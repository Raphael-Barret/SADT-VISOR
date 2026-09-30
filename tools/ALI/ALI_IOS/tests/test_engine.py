"""The engine, with the 2D UNet and the rasterizer stubbed.

What is exercised for real: the model-bundle naming rule, the tooth-label
check, the projection of a predicted mask back onto the mesh, the output tree,
the naming of every file, the run report, and the rule that one mesh failing
must not cost the batch. What is stubbed is the render and the network -- see
`conftest.py` for why the stub is a fixed 2x2 image rather than a mock.
"""

import json
import os

import pytest
import torch

from conftest import tree_of, write_bundle, write_surface
from sadt_ali_common.discovery import WORK_DIRNAME
from sadt_ali_ios import catalog, dispatch, engine, run
from sadt_ali_ios.errors import ToolInputError

BOTH_JAWS = ("Upper_O_model.pth", "Lower_O_model.pth")


def mesh_with(path, teeth, points_per_tooth=3):
    """A mesh carrying exactly these Universal tooth numbers."""
    labels = [number for number in teeth for _ in range(points_per_tooth)]
    return write_surface(path, labels=tuple(labels))


def identify(tmp_path, input_path, bundle, output="out", **kwargs):
    return dispatch.identify(
        input_path=str(input_path),
        model_path=str(bundle),
        output_dir=str(tmp_path / output),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The model bundle, and its one naming rule
# ---------------------------------------------------------------------------

def test_the_published_checkpoint_names_resolve_every_pair(tmp_path):
    bundle = write_bundle(
        tmp_path / "b",
        ["Upper_O_model.pth", "Lower_O_model.pth", "Upper_C_model.pth", "Lower_C_model.pth"],
    )
    weights, unrecognized = engine.discover_weights(bundle)
    assert weights["O"].keys() == {"Upper", "Lower"}
    assert weights["C"].keys() == {"Upper", "Lower"}
    assert unrecognized == []


def test_the_jaw_must_be_named_and_is_never_assumed_upper(tmp_path):
    """The original treated every file not containing "Lower" as upper-jaw
    weights, so a bundle missing its mandibular model quietly predicted the
    lower arch with the maxillary one -- a plausible result, on the wrong
    anatomy, reported as a success."""
    bundle = write_bundle(tmp_path / "b", ["O_model.pth", "Upper_C_model.pth"])
    weights, unrecognized = engine.discover_weights(bundle)
    assert unrecognized == ["O_model.pth"]
    assert "O" not in weights


def test_a_token_is_a_whole_token_not_a_substring(tmp_path):
    """`"Lower" in name` is what the UI did. `Loweralpha_O.pth` is not a
    mandibular checkpoint, and neither is a patient folder called Lowery."""
    bundle = write_bundle(tmp_path / "b", ["Loweralpha_O.pth", "Ocular_Upper.pth"])
    _weights, unrecognized = engine.discover_weights(bundle)
    assert unrecognized == ["Loweralpha_O.pth", "Ocular_Upper.pth"]


def test_the_tokens_are_matched_case_insensitively(tmp_path):
    bundle = write_bundle(tmp_path / "b", ["upper_o_model.pth", "LOWER_O_MODEL.pth"])
    weights, unrecognized = engine.discover_weights(bundle)
    assert weights["O"].keys() == {"Upper", "Lower"}
    assert unrecognized == []


def test_discovery_walks_the_whole_bundle(tmp_path):
    """A published bundle is a folder of folders as often as it is flat."""
    write_bundle(tmp_path / "b" / "occlusal", ["Upper_O_model.pth"])
    write_bundle(tmp_path / "b" / "cervical", ["Upper_C_model.pth"])
    weights, _unrecognized = engine.discover_weights(str(tmp_path / "b"))
    assert set(weights) == {"O", "C"}


def test_a_file_that_is_not_a_checkpoint_is_ignored_rather_than_unrecognised(tmp_path):
    """A bundle ships a README and a licence. Listing them as unrecognised
    weights would bury the one name that really is misspelt."""
    bundle = write_bundle(tmp_path / "b", ["Upper_O_model.pth"])
    (tmp_path / "b" / "README.md").write_text("weights")
    _weights, unrecognized = engine.discover_weights(bundle)
    assert unrecognized == []


def test_the_unrecognised_list_is_sorted(tmp_path):
    bundle = write_bundle(tmp_path / "b", ["zeta.pth", "alpha.pth", "Upper_O.pth"])
    _weights, unrecognized = engine.discover_weights(bundle)
    assert unrecognized == ["alpha.pth", "zeta.pth"]


def test_the_mucogingival_checkpoint_is_recognised_like_any_other(tmp_path):
    """MG is a network code like O and C, so a `Lower_MG_*.pth` resolves
    through the same one rule rather than needing a second one."""
    bundle = write_bundle(tmp_path / "b", ["Lower_MG_model.pth"])
    weights, unrecognized = engine.discover_weights(bundle)
    assert weights["MG"] == {"Lower": os.path.join(bundle, "Lower_MG_model.pth")}
    assert unrecognized == []


def test_a_bundle_that_is_not_a_directory_names_the_basename_only(tmp_path):
    """The message reaches the client verbatim as a 422; the server's own
    directory layout does not travel with it."""
    stray = tmp_path / "secret_job_dir" / "ALI_CBCT_Models.txt"
    os.makedirs(stray.parent)
    stray.write_text("wrong kind")

    with pytest.raises(ToolInputError) as raised:
        engine.discover_weights(str(stray))
    message = str(raised.value)
    assert "ALI_CBCT_Models.txt" in message
    assert "secret_job_dir" not in message


def test_a_bundle_with_no_weights_for_the_selected_networks_is_an_input_error(tmp_path):
    """Nothing the server can do: the caller has to pick the bundle, or the
    networks, that match. So it names both, and lists what it could not read."""
    mesh = mesh_with(tmp_path / "in" / "arch.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "ALI_CBCT_Models", ["Root_Net.pth"])

    with pytest.raises(ToolInputError) as raised:
        engine.predict_landmarks(
            meshes=[(mesh, "arch.vtk")], model_path=bundle, networks=("O",),
            output_dir=str(tmp_path / "out"), device="cpu",
        )
    message = str(raised.value)
    assert "ALI_CBCT_Models" in message
    assert "Occlusal" in message
    assert "Upper_O_model.pth" in message      # the naming rule, spelled out
    assert "Root_Net.pth" in message           # what it could not read


# ---------------------------------------------------------------------------
# Tooth labels: the Crown_Seg handoff
# ---------------------------------------------------------------------------

def test_the_label_check_covers_the_whole_batch_before_any_weight_is_loaded(tmp_path):
    """Discovering this on mesh 40 of 40 costs an hour of inference first."""
    good = write_surface(tmp_path / "in" / "a.vtk")
    raw = write_surface(tmp_path / "in" / "b.vtk", labels=None)
    with pytest.raises(ToolInputError, match="Crown_Seg"):
        engine.require_labels([(good, "a.vtk"), (raw, "b.vtk")])


def test_the_label_check_names_every_array_it_looked_for(tmp_path):
    raw = write_surface(tmp_path / "in" / "b.vtk", labels=None)
    with pytest.raises(ToolInputError) as raised:
        engine.require_labels([(raw, "b.vtk")])
    for name in ("PredictedID", "UniversalID", "Universal_ID"):
        assert name in str(raised.value)


def test_an_unsegmented_batch_is_refused_before_the_engine_starts(tmp_path, stubs):
    write_surface(tmp_path / "in" / "b.vtk", labels=None)
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)
    with pytest.raises(ToolInputError, match="Crown_Seg"):
        identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])
    assert stubs.loaded == []


# ---------------------------------------------------------------------------
# Projecting a predicted mask back onto the mesh
# ---------------------------------------------------------------------------

def test_a_pixel_that_came_from_no_face_is_dropped():
    """`pix_to_face` carries -1 where nothing was rendered. The original passed
    it on, where `faces[-1]` silently selected the mesh's LAST face and dragged
    the landmark's centroid to an arbitrary corner of the arch."""
    logits = torch.zeros(1, 4, 1, 2)
    logits[0, 1, 0, 0] = 1.0
    logits[0, 1, 0, 1] = 1.0
    pix_to_face = torch.tensor([[[[[7.0], [-1.0]]]]])

    assert engine._predicted_faces(logits, pix_to_face, 1) == [7]


def test_the_class_is_taken_from_the_raw_logits_not_a_truncated_copy():
    """The original cast the logits to int16 first, truncating every value
    toward zero -- which turns a 0.6/0.4 win into a 0/0 tie that argmax
    resolves in favour of channel 0, the background. Measured over one Cervical
    pass upstream, that shrinks the landmark channels by 6 283 pixels."""
    logits = torch.zeros(1, 4, 1, 1)
    logits[0, 0, 0, 0] = 0.4
    logits[0, 1, 0, 0] = 0.6
    pix_to_face = torch.tensor([[[[[3.0]]]]])

    assert engine._predicted_faces(logits, pix_to_face, 1) == [3]
    # And under the truncation the port removed, channel 1 would win nothing.
    assert engine._predicted_faces(logits.to(torch.int16), pix_to_face, 1) == []


def test_a_channel_nothing_predicted_yields_no_landmark():
    logits = torch.zeros(1, 4, 1, 1)
    logits[0, 0, 0, 0] = 1.0
    pix_to_face = torch.tensor([[[[[3.0]]]]])
    assert engine._predicted_faces(logits, pix_to_face, 2) == []


# ---------------------------------------------------------------------------
# A run, end to end
# ---------------------------------------------------------------------------

def test_a_run_writes_one_markups_file_per_mesh_plus_the_report(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    mesh_with(tmp_path / "in" / "b.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])

    assert tree_of(str(tmp_path / "out")) == [
        "a_lm_Pred.mrk.json", "b_lm_Pred.mrk.json", "run_report.json",
    ]


def test_the_output_mirrors_the_input_tree(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "siteA" / "p1.vtk", [8, 9])
    mesh_with(tmp_path / "in" / "siteB" / "nested" / "p2.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])

    assert tree_of(str(tmp_path / "out")) == sorted([
        os.path.join("siteA", "p1_lm_Pred.mrk.json"),
        os.path.join("siteB", "nested", "p2_lm_Pred.mrk.json"),
        "run_report.json",
    ])


def test_two_meshes_of_the_same_name_do_not_overwrite_each_other(tmp_path, stubs):
    """The collision the base-name key caused, in the output folder this time:
    both patients were called `scan.vtk` and the second file replaced the
    first, silently."""
    mesh_with(tmp_path / "in" / "patientA" / "scan.vtk", [8, 9])
    mesh_with(tmp_path / "in" / "patientB" / "scan.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])

    assert len(tree_of(str(tmp_path / "out"))) == 3
    assert len(report["cases"]) == 2


def test_the_prediction_id_reaches_the_file_name(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"],
             prediction_ID="Study7")
    assert "a_lm_Study7.mrk.json" in tree_of(str(tmp_path / "out"))


@pytest.mark.parametrize("given", ["", "   ", None])
def test_a_blank_prediction_id_falls_back_to_pred(tmp_path, stubs, given):
    """An empty text field is what a client sends for "I did not fill this in",
    and `scan_lm_.mrk.json` is a file nobody can tell apart from the next run's."""
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"],
                      prediction_ID=given)
    assert report["prediction_ID"] == "Pred"
    assert "a_lm_Pred.mrk.json" in tree_of(str(tmp_path / "out"))


def test_the_output_directory_is_created_when_it_does_not_exist(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    identify(tmp_path, tmp_path / "in", bundle, output="deep/nested/out",
             ios_networks=["Occlusal"])
    assert os.path.isdir(str(tmp_path / "deep" / "nested" / "out"))


def test_nothing_is_written_beside_the_input(tmp_path, stubs, monkeypatch):
    """The original converted DICOM into `<input>/NIFTI/` and wrote a
    segmentation CSV into the extension's own source tree, both of which a
    later run then re-ingested."""
    monkeypatch.chdir(tmp_path)
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)
    before = tree_of(str(tmp_path / "in"))

    identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])

    assert tree_of(str(tmp_path / "in")) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b", "in", "out"]


def test_the_working_directory_does_not_survive_the_run(tmp_path, stubs):
    """A surviving `.ali_work/` means a run crashed -- and it would be shipped
    to the client inside the result archive."""
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])
    assert not os.path.exists(str(tmp_path / "out" / WORK_DIRNAME))


def test_the_working_directory_is_removed_even_when_the_run_fails(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", ["Root_Net.pth"])

    with pytest.raises(ToolInputError):
        identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])
    assert not os.path.exists(str(tmp_path / "out" / WORK_DIRNAME))


def test_run_returns_the_output_directory(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    returned = run(
        input=tmp_path / "in", model=bundle, output_dir=tmp_path / "out",
        networks=["Occlusal"],
    )
    assert str(returned) == str(tmp_path / "out")
    assert os.path.isfile(str(tmp_path / "out" / "run_report.json"))


# ---------------------------------------------------------------------------
# The run report
# ---------------------------------------------------------------------------

def test_the_report_lands_beside_the_results_and_says_what_ran(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "ALI_IOS_Models", BOTH_JAWS)

    identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])
    with open(str(tmp_path / "out" / dispatch.REPORT_NAME), encoding="utf-8") as handle:
        report = json.load(handle)

    assert report["tool"] == "ALI_IOS"
    assert report["mode"] == "IOS"
    assert report["device"] == "cpu"
    assert report["networks"] == ["Occlusal"]
    # So the report says which weights ran even when nobody read the argument.
    assert report["model_bundle"] == "ALI_IOS_Models"
    assert report["summary"] == {"total": 1, "processed": 1, "failed": 0}
    assert isinstance(report["duration_seconds"], float)


def test_the_report_lists_the_landmarks_found_per_scan(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8], points_per_tooth=6)
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])
    scan = report["cases"]["a.vtk"]
    assert scan["status"] == "ok"
    assert scan["input"] == "a.vtk"
    assert scan["landmarks_found"] == sorted(catalog.LABELS["O"]["8"])
    assert scan["landmarks_failed"] == {}


def test_a_requested_network_the_bundle_lacks_is_reported_not_dropped(tmp_path, stubs):
    """"The bundle has no such network" and "the network found nothing on this
    mesh" look identical in the Slicer scene and need opposite fixes: another
    bundle, or another scan."""
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    report = identify(tmp_path, tmp_path / "in", bundle,
                      ios_networks=["Occlusal", "Cervical"])
    assert report["networks"] == ["Occlusal"]
    assert report["landmarks_without_model"] == ["Cervical"]


def test_a_jaw_whose_checkpoint_is_missing_is_reported_not_swallowed(tmp_path, stubs):
    """The original raised a `KeyError` here that was caught and discarded, so
    the jaw simply vanished from the output."""
    mesh_with(tmp_path / "in" / "a.vtk", [8, 19])
    bundle = write_bundle(tmp_path / "b", ["Upper_O_model.pth"])

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])
    assert report["cases"]["a.vtk"]["jaws_without_model"] == {"Occlusal": ["Lower"]}


def test_a_checkpoint_it_could_not_read_is_named_in_the_report(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", list(BOTH_JAWS) + ["Lower_MG_v6.pth", "junk.pth"])

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])
    assert report["models_unrecognized"] == ["junk.pth"]


def test_the_weights_are_loaded_once_per_pass_not_once_per_tooth(tmp_path, stubs):
    """The original instantiated a UNet and called `load_state_dict` INSIDE the
    per-tooth loop -- 28 model loads per scan per network, re-reading the mesh
    every time."""
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9, 10, 11])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])
    assert [code for _path, _device, code in stubs.loaded] == ["O"]


# ---------------------------------------------------------------------------
# Failure, per mesh and per batch
# ---------------------------------------------------------------------------

def test_one_mesh_failing_does_not_cost_the_others(tmp_path, stubs):
    """A batch is a cohort. One patient's mesh being unusable must not throw
    away the thirty-nine that ran."""
    mesh_with(tmp_path / "in" / "good.vtk", [8, 9])
    mesh_with(tmp_path / "in" / "odd.vtk", [77, 88])       # no known tooth number
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])

    assert report["summary"] == {"total": 2, "processed": 1, "failed": 1}
    assert report["cases"]["good.vtk"]["status"] == "ok"
    failed = report["cases"]["odd.vtk"]
    assert failed["status"] == "failed"
    assert "no known tooth number" in failed["error"]
    assert "good_lm_Pred.mrk.json" in tree_of(str(tmp_path / "out"))


def test_one_tooth_failing_does_not_cost_the_rest_of_the_arch(tmp_path, stubs, monkeypatch):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    real = engine._predict_one_tooth

    def flaky(**kwargs):
        if kwargs["tooth_number"] == 8:
            raise RuntimeError("rasterizer said no")
        return real(**kwargs)

    monkeypatch.setattr(engine, "_predict_one_tooth", flaky)
    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])

    scan = report["cases"]["a.vtk"]
    assert scan["status"] == "ok"
    assert scan["landmarks_failed"] == {
        "Upper-8": "RuntimeError: rasterizer said no"
    }
    assert scan["landmarks_found"]


def test_a_batch_where_nothing_worked_raises_naming_the_first_error(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [77, 88])
    mesh_with(tmp_path / "in" / "b.vtk", [77, 88])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    with pytest.raises(RuntimeError, match="no known tooth number"):
        identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"])


def test_an_empty_network_selection_names_the_argument_to_fill_in(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    with pytest.raises(ToolInputError) as raised:
        identify(tmp_path, tmp_path / "in", bundle, ios_networks=[])
    message = str(raised.value)
    # The name a CALLER uses, which is the one `run()` publishes -- `networks`.
    # `ios_networks` is the engine's own keyword and never leaves this package;
    # naming it in a 422 would send someone looking for an argument the schema
    # does not offer. Read off the signature rather than written down, so the
    # message and the published name cannot drift apart again.
    import inspect

    from sadt_ali_ios import run as published

    name = next(p for p in inspect.signature(published).parameters if "networks" in p)
    assert name == "networks"
    assert f"'{name}'" in message
    assert "Occlusal" in message and "Mucogingival" in message


def test_an_unknown_network_is_refused_rather_than_dropped(tmp_path, stubs):
    """`Literal` is published, not enforced -- the runner calls `run(**params)`
    from a JSON object. A stale client naming a family that no longer exists
    has to be told, not handed a narrower run than it asked for."""
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    with pytest.raises(ValueError, match="Buccal"):
        identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Buccal"])


# ---------------------------------------------------------------------------
# Mucogingival: the jaw restriction, the arch fit and the forced point
# ---------------------------------------------------------------------------

LOWER_BUNDLE = ("Lower_MG_model.pth", "Upper_O_model.pth", "Lower_O_model.pth")


def test_mucogingival_on_a_maxilla_produces_nothing_and_fails_nothing(tmp_path, stubs):
    """MG was trained on the mandible alone, so an upper arch is not a missing
    model -- it is a question the network cannot be asked. The occlusal pass
    must be unaffected and `jaws_without_model` must stay empty."""
    mesh_with(tmp_path / "in" / "upper.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", LOWER_BUNDLE)

    report = identify(tmp_path, tmp_path / "in", bundle,
                      ios_networks=["Occlusal", "Mucogingival"])
    scan = report["cases"]["upper.vtk"]

    assert scan["status"] == "ok"
    assert scan["jaws_without_model"] == {}
    assert all(not name.endswith("MG") for name in scan["landmarks_found"])


def test_a_missing_mucogingival_checkpoint_is_reported_against_the_lower_jaw(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "lower.vtk", list(catalog.MG_TEETH))
    bundle = write_bundle(tmp_path / "b", ["Upper_MG_model.pth", "Lower_O_model.pth"])

    report = identify(tmp_path, tmp_path / "in", bundle,
                      ios_networks=["Occlusal", "Mucogingival"])
    assert report["cases"]["lower.vtk"]["jaws_without_model"] == {"Mucogingival": ["Lower"]}


def test_a_mucogingival_run_places_the_positional_names(tmp_path, stubs):
    """Six MG output names collide with the TRAINING name of a different tooth,
    so the table is positional. Deriving `<tooth><type>` here would mislabel
    half the arch."""
    mesh_with(tmp_path / "in" / "lower.vtk", list(catalog.MG_TEETH))
    bundle = write_bundle(tmp_path / "b", ["Lower_MG_model.pth"])

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Mucogingival"])
    found = report["cases"]["lower.vtk"]["landmarks_found"]
    assert found == sorted(catalog.MG_OUTPUT_NAME)
    assert "L0MG" in found and "LR1MG" in found


def test_a_forced_mucogingival_point_carries_its_caveat_into_the_file(tmp_path, stubs):
    """A tooth that predicts no pixel at all falls back to its most likely
    ones rather than being dropped -- upstream measures ~5 mm of error against
    ~1.2 mm for a won point, which is worth having as a point to be REVIEWED
    rather than a hole in the mucogingival line."""
    stubs.mg_class_of_pixel = ((0, 0), (0, 0))
    mesh_with(tmp_path / "in" / "lower.vtk", list(catalog.MG_TEETH))
    bundle = write_bundle(tmp_path / "b", ["Lower_MG_model.pth"])

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Mucogingival"])
    degraded = report["cases"]["lower.vtk"]["landmarks_degraded"]
    assert set(degraded) == set(catalog.MG_OUTPUT_NAME)
    assert all(note.startswith("forced (confidence") for note in degraded.values())

    with open(str(tmp_path / "out" / "lower_lm_Pred.mrk.json"), encoding="utf-8") as handle:
        written = json.load(handle)
    descriptions = {
        point["label"]: point["description"]
        for point in written["markups"][0]["controlPoints"]
    }
    assert descriptions["L0MG"].startswith("forced (confidence")


def test_a_tooth_the_segmentation_missed_is_aimed_from_the_arch_and_says_so(tmp_path, stubs):
    """A gap in the segmentation must not become a gap in the mucogingival
    line. The point is placed from a fit of the arch through the teeth that
    ARE labelled, and carries a caveat saying so."""
    present = [number for number in catalog.MG_TEETH if number != 25]
    mesh_with(tmp_path / "in" / "lower.vtk", present)
    bundle = write_bundle(tmp_path / "b", ["Lower_MG_model.pth"])

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Mucogingival"])
    scan = report["cases"]["lower.vtk"]

    assert "L0MG" in scan["landmarks_found"]
    assert scan["landmarks_degraded"]["L0MG"] == (
        "cameras aimed from an arch fit, tooth not segmented"
    )


def test_a_won_mucogingival_point_carries_no_caveat_at_all(tmp_path, stubs):
    """`landmarks_degraded` is only present when something was degraded: an
    always-present key of empty notes would train the reader to ignore it."""
    mesh_with(tmp_path / "in" / "lower.vtk", list(catalog.MG_TEETH))
    bundle = write_bundle(tmp_path / "b", ["Lower_MG_model.pth"])

    report = identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Mucogingival"])
    assert "landmarks_degraded" not in report["cases"]["lower.vtk"]


# ---------------------------------------------------------------------------
# The lazy stack
# ---------------------------------------------------------------------------

def test_the_whole_lazy_stack_is_imported_once_before_any_mesh_is_touched(monkeypatch):
    """A missing dependency belongs to the venv, not to a patient's mesh. The
    per-mesh `except` would otherwise report it once per mesh and then hide it
    behind "produced no landmarks for any mesh"."""
    from sadt_ali_ios import render as render_module
    from sadt_ali_ios import surface as surface_module

    called = []
    monkeypatch.setattr(engine, "import_torch", lambda: called.append("torch"))
    monkeypatch.setattr(surface_module, "import_vtk", lambda: called.append("vtk"))
    monkeypatch.setattr(render_module, "import_pytorch3d", lambda: called.append("pytorch3d"))
    monkeypatch.setattr(engine, "_import_unet", lambda: called.append("monai"))

    engine.check_dependencies()
    assert called == ["torch", "vtk", "pytorch3d", "monai"]


# ---------------------------------------------------------------------------
# The device
# ---------------------------------------------------------------------------

def test_the_device_is_resolved_once_from_the_argument(monkeypatch):
    """The original asked `torch.cuda.is_available()` independently in five
    modules, so a run that asked for CPU still used a card that happened to be
    present."""
    from sadt_ali_ios import torch_helpers

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_helpers.resolve_device("cpu") == "cpu"
    assert torch_helpers.resolve_device("cuda") == "cuda"
    # Whitespace and case are what a form field sends.
    assert torch_helpers.resolve_device("  CUDA  ") == "cuda"


def test_cuda_falls_back_to_cpu_when_no_card_is_visible(monkeypatch, caplog):
    """"cuda" is the published default, so a CPU-only deployment must run
    rather than fail -- and must say why it is slow."""
    from sadt_ali_ios import torch_helpers

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        assert torch_helpers.resolve_device("cuda") == "cpu"
    assert "CUDA is unavailable" in caplog.text


def test_an_omitted_device_is_cpu_rather_than_whatever_is_plugged_in(monkeypatch):
    """`run()` always passes one, so this is the direct-API path -- and
    defaulting to a card there would make a library call grab the GPU."""
    from sadt_ali_ios import torch_helpers

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_helpers.resolve_device(None) == "cpu"


def test_the_resolved_device_is_what_the_checkpoints_are_loaded_onto(tmp_path, stubs):
    mesh_with(tmp_path / "in" / "a.vtk", [8, 9])
    bundle = write_bundle(tmp_path / "b", BOTH_JAWS)

    identify(tmp_path, tmp_path / "in", bundle, ios_networks=["Occlusal"], device="cuda")
    assert {device for _path, device, _code in stubs.loaded} == {"cpu"}


# ---------------------------------------------------------------------------
# Projecting a mask back onto the mesh, and the order it is done in
# ---------------------------------------------------------------------------

def test_a_predicted_mask_is_read_in_row_major_order_with_nothing_rendered_dropped():
    """`_predicted_faces` walks (view, row, column) in that order, and a pixel
    where nothing was rendered is dropped rather than selecting the last face.

    Both halves are pinned because both were lost once. The original passed
    `-1` on, where `faces[-1]` silently selected the mesh's last face and
    dragged the centroid to a corner of the arch; and the order is what the
    caller's sequential float32 sum depends on.
    """
    # Two views, two classes, 1x2 pixels. Class 1 wins everywhere except the
    # last pixel, so three pixels are claimed, in reading order.
    predictions = torch.tensor(
        [
            [[[0.0, 0.0]], [[1.0, 1.0]]],   # view 0: both pixels class 1
            [[[0.0, 1.0]], [[1.0, 0.0]]],   # view 1: first class 1, second class 0
        ]
    )
    # float32 on purpose: this is what the tensor the views are accumulated
    # into actually holds, `torch.cat` having promoted int64 to float.
    pix_to_face = torch.tensor(
        [
            [[[[70.0], [-1.0]]]],           # view 0: face 70, then nothing
            [[[[42.0], [99.0]]]],           # view 1: face 42, then face 99
        ]
    )

    # View 0 pixel 0 -> 70, view 0 pixel 1 -> dropped, view 1 pixel 0 -> 42.
    assert engine._predicted_faces(predictions, pix_to_face, 1) == [70, 42]
    # Class 0 wins only at view 1 pixel 1.
    assert engine._predicted_faces(predictions, pix_to_face, 0) == [99]
    assert engine._predicted_faces(predictions, pix_to_face, 2) == []


class _RecordingLocator:
    """A point locator that keeps the centroid it was asked about."""

    def __init__(self):
        self.asked = []

    def FindClosestPoint(self, point):
        self.asked.append(point)
        return 0


class _OnePointSurface:
    def GetPoint(self, point_id):
        return (0.0, 0.0, 0.0)


def test_the_centroid_is_accumulated_in_order_rather_than_as_a_tree():
    """A landmark is the mesh point nearest the centroid of the faces a mask
    claimed, and that centroid must be the number the original loop produced.

    float32 addition is not associative, so `sum(dim=0)` -- a tree reduction --
    is a DIFFERENT number from adding the rows one after another, by around a
    ULP per thousand rows. It is a small number, and it is a number that decides
    which of two adjacent vertices a landmark lands on. This pins the scan:
    swap `cumsum(dim=0)[-1]` for `sum(dim=0)` and this test fails.
    """
    torch.manual_seed(0)
    # Coordinates of the size the engine works in: the mesh is scaled into the
    # unit sphere before anything is rendered.
    vertices = ((torch.rand(4000, 3) - 0.5) * 2.0).unsqueeze(0)
    face_table = torch.arange(3999).reshape(-1, 3).unsqueeze(0)
    faces = list(range(face_table.shape[1]))

    locator = _RecordingLocator()
    engine._landmark_position(faces, face_table, vertices, locator, _OnePointSurface())

    vertex_ids = [int(face_table[0][face][corner]) for face in faces for corner in range(3)]
    sequential = sum(vertices[0][vertex_id] for vertex_id in vertex_ids) / len(vertex_ids)
    assert locator.asked[0].tobytes() == sequential.numpy().tobytes()


def test_a_mask_that_claimed_no_face_has_no_position():
    """An empty list, not an exception and not the origin: the caller reads
    `None` as "this landmark was not found on this tooth"."""
    assert engine._landmark_position(
        [], torch.zeros(1, 1, 3, dtype=torch.int64), torch.zeros(1, 3, 3),
        _RecordingLocator(), _OnePointSurface()
    ) is None

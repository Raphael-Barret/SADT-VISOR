"""Unit tests for the ALI tool.

No GPU, no real model weights and no network inference: the agent that walks
the volume is stubbed, so everything AROUND it -- mode detection, DICOM
recognition, input discovery, weight discovery, the landmark vocabulary,
output naming and tree preservation, the run report, and every cross-argument
rule -- is exercised for real.

The parts that genuinely cannot run here are skipped rather than faked: the
IOS engine needs pytorch3d, which publishes no usable wheel and is compiled
from source behind the `ios` extra. What does not need it (the model naming
rule, the tooth vocabulary, the network selection, the tooth-label check) is
tested regardless, since that is where the original's defects were.

    cd tools/ALI && uv run pytest
"""

import json
import os
import typing

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_ali_cbct import dispatch, run
from sadt_ali_common import markups
from sadt_ali_cbct import catalog as cbct_catalog
from sadt_ali_cbct.errors import ToolInputError, ToolUnavailableError



ALL_REGIONS = list(cbct_catalog.REGION_NAMES)
CRANIAL_BASE_ONLY = ["Cranial base"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_volume(path, size=(16, 16, 16)):
    """A small synthetic CBCT with a bright cube in it."""
    array = np.zeros(size[::-1], dtype=np.int16)
    array[4:12, 4:12, 4:12] = 800
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 0.5))
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    sitk.WriteImage(image, str(path))
    return str(path)


def write_surface(path, labelled=True):
    """A minimal .vtk polydata, optionally carrying a tooth-label array."""
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
        points.InsertNextPoint(*coordinates)

    polys = vtk.vtkCellArray()
    for triangle in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
        polys.InsertNextCell(3)
        for point_id in triangle:
            polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName("Universal_ID")
        for value in (8, 8, 8, 8):
            labels.InsertNextValue(value)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


def write_cbct_bundle(root, landmarks_by_region):
    """A CBCT bundle in the <region>/<landmark>/<scale>/*.pth layout."""
    for region, labels in landmarks_by_region.items():
        for label in labels:
            for scale in cbct_catalog.SCALE_KEYS:
                folder = root / region / label / scale
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{label}_Net_{scale}.pth").write_bytes(b"fake checkpoint")
    return str(root)


def write_ios_bundle(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"fake checkpoint")
    return str(root)


@pytest.fixture
def stub_agent(monkeypatch):
    """Replace the deep-RL search with a deterministic voxel position.

    Every landmark whose name starts with "X" is reported as not found, so the
    report's two failure kinds can be told apart in a test.
    """
    from sadt_ali_cbct import agent as agent_module
    from sadt_ali_cbct import engine as cbct_engine

    class StubBrain:
        def __init__(self, scale_keys, device, out_channels=6):
            self.scale_keys = scale_keys

        def load(self, weights_per_scale):
            assert set(weights_per_scale) == set(self.scale_keys)

        def release(self):
            pass

    class StubAgent:
        def __init__(self, target, scale_keys, brain, environment, **_kwargs):
            self.target = target
            self.environment = environment

        def search(self, max_seconds):
            if self.target.startswith("X"):
                raise agent_module.NotFound("stubbed failure")
            return np.array([3.0, 4.0, 5.0])

    monkeypatch.setattr(cbct_engine, "Brain", StubBrain)
    monkeypatch.setattr(cbct_engine, "Agent", StubAgent)
    monkeypatch.setattr(cbct_engine, "resolve_device", lambda requested=None: "cpu")


@pytest.fixture
def cbct_environment():
    pytest.importorskip("monai")
    pytest.importorskip("itk")
    pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# The landmark vocabulary
# ---------------------------------------------------------------------------

def test_impacted_canine_spellings_both_resolve():
    """The defect that lost a whole patient's landmarks.

    The Slicer UI called them UR3OI/UL3OI/UR3RI/UL3RI and the CLI's tables
    called them UR3OIP/.../UL3RIP. `LABEL_GROUPS[landmark]` was then indexed
    with no guard inside the save loop, so one unknown name raised a KeyError
    that was caught far above -- and NOTHING was written for that scan,
    including every landmark that had been found correctly.
    """
    for ui_name, canonical_name in (
        ("UR3OI", "UR3OIP"),
        ("UL3OI", "UL3OIP"),
        ("UR3RI", "UR3RIP"),
        ("UL3RI", "UL3RIP"),
    ):
        assert cbct_catalog.canonical(ui_name) == canonical_name
        assert cbct_catalog.group_of(ui_name) == "CI"
        assert cbct_catalog.group_of(canonical_name) == "CI"


def test_group_of_never_raises_on_an_unknown_name():
    assert cbct_catalog.group_of("NoSuchLandmark") == cbct_catalog.UNGROUPED


def test_scale_keys_match_the_shipped_weight_folders():
    assert cbct_catalog.SCALE_KEYS == ("1", "0-3")


def test_region_codes_from_a_selection():
    assert cbct_catalog.region_codes(["Cranial base", "Lower"]) == ("CB", "L")
    # The codes are accepted too, so a caller driving this package directly
    # need not know the display spellings.
    assert cbct_catalog.region_codes(["CB", "L"]) == ("CB", "L")
    # An omitted optional argument must fall back to every region, not to none.
    assert cbct_catalog.region_codes(None) == cbct_catalog.REGION_CODES
    # Declaration order, not the caller's.
    assert cbct_catalog.region_codes(["Lower", "Cranial base"]) == ("CB", "L")


def test_an_unknown_region_is_refused_not_dropped():
    """`Literal` is published, not enforced -- the runner calls run(**params)
    from a JSON object. A stale client naming a region that no longer exists
    has to be told, not handed a narrower run than it asked for."""
    with pytest.raises(ValueError, match="Cranial base"):
        cbct_catalog.region_codes(["Cranial base", "Sagittal"])








# ---------------------------------------------------------------------------
# The published options and the catalogs cannot drift apart
# ---------------------------------------------------------------------------

def _choices(argument):
    """The `Literal` options `run()` publishes for one argument."""
    hint = typing.get_type_hints(run)[argument]
    if typing.get_origin(hint) is list:
        hint = typing.get_args(hint)[0]
    return list(typing.get_args(hint))


def test_the_published_regions_are_the_catalogs_own():
    """`Literal` takes literals only, so it cannot be built from the catalog.
    That makes the signature a second declaration of the same set, and this is
    what keeps the two honest: a region added to one and not the other would be
    unselectable from the client, or offered and then refused."""
    assert _choices("regions") == list(cbct_catalog.REGION_NAMES)


def test_the_published_landmarks_are_the_catalogs_own():
    assert _choices("landmarks") == list(cbct_catalog.LABELS)




def test_every_published_default_is_one_of_its_own_options():
    """A default outside its option list gives the client a picker that cannot
    produce the value the tool starts from."""
    import inspect

    for argument in ("regions", "landmarks", "device"):
        default = inspect.signature(run).parameters[argument].default
        options = _choices(argument)
        for value in (default if isinstance(default, list) else [default]):
            assert value in options, (argument, value)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def test_a_single_volume_is_cbct(tmp_path):
    scan = write_volume(tmp_path / "in" / "patient01.nii.gz")
    detected = dispatch.detect(scan, str(tmp_path / "work"))
    assert detected.mode == dispatch.CBCT
    assert detected.scans == [(scan, "patient01.nii.gz")]


def test_a_single_surface_is_ios(tmp_path):
    mesh = write_surface(tmp_path / "in" / "arch.vtk")
    detected = dispatch.detect(mesh, str(tmp_path / "work"))
    assert detected.mode == dispatch.IOS


def test_a_folder_of_volumes_is_cbct_and_keeps_its_tree(tmp_path):
    """A folder says nothing about which engine applies -- which is exactly why
    there is no `mode` argument for the caller to get wrong.

    The server unpacks a `.zip` before `run()` is called, so what arrives here
    is always a real directory; this is the shape it arrives in.
    """
    write_volume(tmp_path / "cohort" / "siteA" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "siteB" / "patient01.nii.gz")

    detected = dispatch.detect(str(tmp_path / "cohort"), str(tmp_path / "work"))
    assert detected.mode == dispatch.CBCT
    # Two patients with the SAME base name in different folders. Keyed by
    # relative path, they stay distinct; the original keyed by file.name and
    # one silently replaced the other.
    assert sorted(key for _path, key in detected.scans) == [
        os.path.join("siteA", "patient01.nii.gz"),
        os.path.join("siteB", "patient01.nii.gz"),
    ]


def test_a_mixed_input_is_refused_rather_than_guessed(tmp_path):
    write_volume(tmp_path / "mixed" / "patient01.nii.gz")
    write_surface(tmp_path / "mixed" / "arch.vtk")

    with pytest.raises(ToolInputError) as raised:
        dispatch.detect(str(tmp_path / "mixed"), str(tmp_path / "work"))
    assert "mixes" in str(raised.value)


def test_an_input_with_nothing_recognizable_says_what_it_wanted(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("nothing here")

    with pytest.raises(ToolInputError) as raised:
        dispatch.detect(str(tmp_path / "empty"), str(tmp_path / "work"))
    # Volume extensions only: this tool no longer offers the surface half, so
    # listing .vtk here would advertise something it refuses.
    assert ".nii.gz" in str(raised.value)


def test_detection_is_recursive(tmp_path):
    write_volume(tmp_path / "deep" / "a" / "b" / "c" / "patient.nii.gz")
    detected = dispatch.detect(str(tmp_path / "deep"), str(tmp_path / "work"))
    assert len(detected.scans) == 1


def test_the_working_directory_is_not_rediscovered_as_input(tmp_path):
    """`output_dir` may legitimately be the input folder, and `.ali_work/`
    lives inside it. Its converted DICOM must not come back round as scans."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    stale = tmp_path / "cohort" / dispatch.WORK_DIRNAME / "dicom_converted"
    write_volume(stale / "leftover.nii.gz")

    detected = dispatch.detect(str(tmp_path / "cohort"), str(tmp_path / "work"))
    assert [key for _path, key in detected.scans] == ["patient01.nii.gz"]


def test_a_folder_of_dicom_is_not_mistaken_for_an_empty_input(tmp_path, monkeypatch):
    """DICOM slices carry no extension, so only GDCM can recognize them --
    which is why the client offers a folder picker and why detection probes."""
    from sadt_ali_cbct import preprocess

    series = tmp_path / "cohort" / "patient01"
    series.mkdir(parents=True)
    for index in range(3):
        (series / f"IM{index:06d}").write_bytes(b"not really dicom")

    monkeypatch.setattr(preprocess, "is_dicom_series", lambda directory: True)
    monkeypatch.setattr(
        preprocess,
        "convert_dicom_series",
        lambda directory, destination: write_volume(destination),
    )

    detected = dispatch.detect(str(tmp_path / "cohort"), str(tmp_path / "work"))
    assert detected.mode == dispatch.CBCT
    assert detected.converted_dicom == 1
    # Converted into the WORKING dir, never into the user's own folder -- the
    # original wrote <input>/NIFTI/ and then re-ingested it on the next run.
    converted_path = detected.scans[0][0]
    assert str(tmp_path / "work") in converted_path
    assert not (tmp_path / "cohort" / "NIFTI").exists()


def test_a_folder_already_holding_volumes_is_not_probed_for_dicom(tmp_path, monkeypatch):
    from sadt_ali_cbct import preprocess

    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    probed = []
    monkeypatch.setattr(
        preprocess, "is_dicom_series", lambda directory: probed.append(directory) or False
    )

    dispatch.detect(str(tmp_path / "cohort"), str(tmp_path / "work"))
    assert probed == []


# ---------------------------------------------------------------------------
# CBCT model bundles
# ---------------------------------------------------------------------------

def test_weight_discovery_reads_the_folder_tree(tmp_path):
    from sadt_ali_cbct import engine as cbct_engine

    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba", "S"]})
    weights = cbct_engine.discover_weights(bundle)

    assert sorted(weights) == ["Ba", "S"]
    assert sorted(weights["Ba"]) == sorted(cbct_catalog.SCALE_KEYS)


def test_a_landmark_missing_one_scale_is_not_offered(tmp_path):
    """Half a bundle must be reported up front, not fail mid-run: the agent
    walks the coarse scale and then the fine one, and needs both."""
    from sadt_ali_cbct import engine as cbct_engine

    folder = tmp_path / "bundle" / "Ba" / "1"
    folder.mkdir(parents=True)
    (folder / "Ba_Net_1.pth").write_bytes(b"fake")

    assert cbct_engine.discover_weights(str(tmp_path / "bundle")) == {}


def test_requested_landmarks_separates_missing_from_unknown(tmp_path):
    from sadt_ali_cbct import engine as cbct_engine

    weights = {"Ba": {}, "S": {}, "Mystery": {}}
    runnable, without_model, ungrouped = cbct_engine.requested_landmarks(weights, ["CB"])

    assert runnable == ("Ba", "S")
    # Every other cranial-base landmark the catalog knows about: the client
    # renders this as "not in the selected model bundle", i.e. "use another".
    assert "N" in without_model and "Ba" not in without_model
    # Weights for a landmark this catalog has never heard of are surfaced,
    # not silently ignored.
    assert ungrouped == ["Mystery"]


def test_a_bundle_using_the_ui_spelling_resolves(tmp_path):
    from sadt_ali_cbct import engine as cbct_engine

    bundle = write_cbct_bundle(tmp_path / "bundle", {"Canine": ["UR3OI"]})
    weights = cbct_engine.discover_weights(bundle)

    assert list(weights) == ["UR3OIP"]
    runnable, _missing, ungrouped = cbct_engine.requested_landmarks(weights, ["CI"])
    assert runnable == ("UR3OIP",)
    assert ungrouped == []


def test_a_bundle_of_the_wrong_kind_is_an_input_error(tmp_path, cbct_environment):
    """Naming the IOS bundle for a CBCT run (or vice versa) has to answer with
    a message the client shows verbatim, not a stack trace in a log."""
    from sadt_ali_cbct import engine as cbct_engine

    ios_bundle = write_ios_bundle(tmp_path / "ios", ["Upper_O_model.pth"])
    with pytest.raises(ToolInputError, match="No CBCT landmark weights"):
        cbct_engine.predict_landmarks(
            scans=[],
            model_path=ios_bundle,
            regions=None,
            prediction_ID="Pred",
            output_dir=str(tmp_path / "out"),
            work_dir=str(tmp_path / "work"),
        )


# ---------------------------------------------------------------------------
# IOS model bundles and the tooth-label precondition
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# The markups writer
# ---------------------------------------------------------------------------

def test_markups_are_written_as_a_slicer_file(tmp_path):
    path = markups.write(
        {"Ba": np.array([1.5, 2.5, 3.5]), "S": np.array([4.0, 5.0, 6.0])},
        str(tmp_path / "out" / f"scan_lm_Pred{markups.MARKUPS_EXTENSION}"),
    )
    assert path.endswith(".mrk.json")

    content = json.loads(open(path, encoding="utf-8").read())
    node = content["markups"][0]
    assert node["coordinateSystem"] == "LPS"
    assert [point["label"] for point in node["controlPoints"]] == ["Ba", "S"]
    # numpy scalars are cast to float on the way in: json.dump cannot
    # serialize them, and the failure would land after all the inference.
    assert node["controlPoints"][0]["position"] == [1.5, 2.5, 3.5]


def test_both_engines_write_the_same_extension():
    """CBCT wrote .mrk.json and IOS wrote .json for byte-identical content,
    and Slicer only associates the first with a markups node."""
    assert markups.MARKUPS_EXTENSION == ".mrk.json"


def test_the_display_node_is_visible(tmp_path):
    """Regression, and the nastiest kind: the file is perfectly valid, Slicer
    loads it, the node appears in the Markups module -- and nothing is drawn.

    Both original CLIs wrote `display.visibility: false`. That is independent
    of each control point's own `visibility`, so the points below are visible
    inside a node that is not displayed. Dropping a result file into Slicer
    showed an empty scene.
    """
    path = markups.write({"Ba": (1.0, 2.0, 3.0)}, str(tmp_path / "scan.mrk.json"))
    node = json.loads(open(path, encoding="utf-8").read())["markups"][0]

    assert node["display"]["visibility"] is True
    assert node["controlPoints"][0]["visibility"] is True
    # Off by design, and not the same kind of statement as the two above: a
    # point shows on the slice it is on, nothing more. Projecting 113 of them
    # onto neighbouring slices was tried and crowds the view used to judge
    # placement. It stays one checkbox away in the Markups module.
    assert node["display"]["sliceProjection"] is False


def test_positions_are_plain_json_floats(tmp_path):
    """numpy scalars are not JSON-serializable, and both engines produce
    coordinates as numpy arrays. The cast has to happen in the writer."""
    path = markups.write(
        {"Ba": np.array([1.5, 2.5, 3.5], dtype=np.float32)},
        str(tmp_path / "scan.mrk.json"),
    )
    position = json.loads(open(path, encoding="utf-8").read())[
        "markups"
    ][0]["controlPoints"][0]["position"]

    assert all(type(value) is float for value in position)


# ---------------------------------------------------------------------------
# End to end through run(), agent stubbed
# ---------------------------------------------------------------------------

def test_a_cbct_run_writes_one_file_per_scan(tmp_path, stub_agent, cbct_environment):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba", "S", "N"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        regions=CRANIAL_BASE_ONLY,
    )

    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())
    assert report["mode"] == "CBCT"
    assert report["summary"] == {"total": 1, "processed": 1, "failed": 0}

    produced = sorted(output_dir.rglob("*.mrk.json"))
    # ONE file holding every region, not one per anatomical group -- which is
    # what every downstream tool (ASO, AREG, AutoMatrix) had to recombine.
    assert len(produced) == 1
    assert produced[0].name == "patient01_lm_Pred.mrk.json"

    content = json.loads(produced[0].read_text())
    assert sorted(point["label"] for point in content["markups"][0]["controlPoints"]) == [
        "Ba", "N", "S"
    ]


def test_run_returns_the_output_directory(tmp_path, stub_agent, cbct_environment):
    """The contract describe.py publishes as `"returns": "path"`."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    returned = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        regions=CRANIAL_BASE_ONLY,
    )
    assert returned == tmp_path / "out"
    assert returned.is_dir()


def test_nothing_is_written_beside_the_input(tmp_path, stub_agent, cbct_environment):
    """Everything goes under `output_dir`, which the caller owns."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})
    before = sorted(p for p in (tmp_path / "cohort").rglob("*") if p.is_file())

    run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        regions=CRANIAL_BASE_ONLY,
    )

    assert sorted(p for p in (tmp_path / "cohort").rglob("*") if p.is_file()) == before


def test_the_working_directory_does_not_survive_the_run(tmp_path, stub_agent, cbct_environment):
    """A leftover `.ali_work/` means a run crashed, so it has to be reliable."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        regions=CRANIAL_BASE_ONLY,
    )
    assert not (output_dir / dispatch.WORK_DIRNAME).exists()


def test_the_working_directory_is_removed_even_when_the_run_fails(tmp_path, cbct_environment):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    empty_bundle = tmp_path / "bundle"
    empty_bundle.mkdir()

    with pytest.raises(ToolInputError):
        run(
            input=tmp_path / "cohort",
            model=empty_bundle,
            output_dir=tmp_path / "out",
            regions=CRANIAL_BASE_ONLY,
        )
    assert not (tmp_path / "out" / dispatch.WORK_DIRNAME).exists()


def test_a_batch_keeps_its_tree_so_homonyms_cannot_collide(
    tmp_path, stub_agent, cbct_environment
):
    write_volume(tmp_path / "cohort" / "siteA" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "siteB" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        regions=CRANIAL_BASE_ONLY,
    )

    produced = sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*.mrk.json"))
    assert produced == [
        os.path.join("siteA", "patient01_lm_Pred.mrk.json"),
        os.path.join("siteB", "patient01_lm_Pred.mrk.json"),
    ]


def test_the_report_tells_a_missing_model_from_a_failed_search(
    tmp_path, stub_agent, cbct_environment
):
    """The two failures look identical in the Slicer scene -- a landmark that
    is simply not there -- and need opposite fixes: another bundle, or another
    scan. The report is the only place they are distinguishable."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    # "XBa" is not in the catalog but IS in the bundle, so it is reported as
    # ungrouped and never run; "S" is in the catalog but not in the bundle.
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})
    for scale in cbct_catalog.SCALE_KEYS:
        folder = tmp_path / "bundle" / "Cranial_Base" / "XBa" / scale
        folder.mkdir(parents=True)
        (folder / "x.pth").write_bytes(b"fake")

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        regions=CRANIAL_BASE_ONLY,
    )
    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())

    scan = report["cases"]["patient01.nii.gz"]
    assert scan["landmarks_found"] == ["Ba"]
    assert "S" in report["landmarks_without_model"]
    assert report["landmarks_ungrouped"] == ["XBa"]
    assert scan["landmarks_failed"] == {}


def test_a_landmark_that_never_converges_does_not_cost_the_others(
    tmp_path, stub_agent, cbct_environment
):
    """The unguarded `LABEL_GROUPS[landmark]` lookup meant one bad landmark
    lost the whole patient, including the ones already found."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})
    # The stub fails any landmark starting with "X". "XN" is aliased into the
    # catalog below so it is genuinely requested and genuinely runs.
    for scale in cbct_catalog.SCALE_KEYS:
        folder = tmp_path / "bundle" / "Cranial_Base" / "XN" / scale
        folder.mkdir(parents=True)
        (folder / "x.pth").write_bytes(b"fake")
    cbct_catalog.LABEL_GROUPS["XN"] = "CB"
    try:
        output_dir = run(
            input=tmp_path / "cohort",
            model=bundle,
            output_dir=tmp_path / "out",
            regions=CRANIAL_BASE_ONLY,
        )
    finally:
        del cbct_catalog.LABEL_GROUPS["XN"]

    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())
    scan = report["cases"]["patient01.nii.gz"]
    assert scan["status"] == "ok"
    assert scan["landmarks_found"] == ["Ba"]
    assert "XN" in scan["landmarks_failed"]
    assert scan["produced"]


def test_the_run_report_lands_beside_the_results(tmp_path, stub_agent, cbct_environment):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort", model=bundle, output_dir=tmp_path / "out"
    )

    # The Slicer module reads exactly this file, by this name.
    on_disk = output_dir / dispatch.REPORT_NAME
    assert on_disk.is_file()
    written = json.loads(on_disk.read_text())
    assert written["mode"] == "CBCT"
    assert written["regions"]
    # So a result says which weights produced it.
    assert written["model_bundle"] == "bundle"


def test_the_marker_is_constant_and_is_what_the_tool_publishes(
        tmp_path, stub_agent, cbct_environment):
    """It used to be an argument, `prediction_ID`, defaulting to "Pred".

    That made the marker a property of the REQUEST, so nothing downstream
    could predict it -- and pairing a scan with its landmarks, or working out
    which results belong to one patient, both have to strip a marker they can
    predict. A caller wanting to label a run labels the output FOLDER.

    Asserted against `OUTPUT_SUFFIXES` rather than against the string, so the
    file this tool writes and the marker it publishes cannot drift apart.
    """
    from sadt_ali_cbct import OUTPUT_SUFFIXES

    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
    )
    written = [p.name for p in output_dir.rglob("*.mrk.json")]
    assert written == ["patient01_lm_Pred.mrk.json"]
    assert any(name.startswith("patient01" + suffix.rstrip("_"))
               or suffix in name for name in written
               for suffix in OUTPUT_SUFFIXES), (
        "the marker written and the marker published have drifted apart"
    )


# ---------------------------------------------------------------------------
# `search_steps` -- a budget in forward passes, not in seconds
# ---------------------------------------------------------------------------

def test_the_search_budget_is_one_number_for_every_device():
    """0 means "not specified": there is no nullable type in the schema, so the
    argument cannot default to None the way the setting it replaces did.

    And it takes no device, which is the point of counting steps: seconds were
    a property of the machine and of how busy it was, so the same request was
    a different search on a contended card. A step is a forward pass.
    """
    from sadt_ali_cbct import engine as cbct_engine

    assert cbct_engine.search_budget(0) == cbct_engine._DEFAULT_SEARCH_STEPS
    assert cbct_engine.search_budget(250) == 250
    # Comfortably above the worst search measured on the reference scan (193
    # steps), or a landmark that used to be found would start being lost.
    assert cbct_engine._DEFAULT_SEARCH_STEPS > 4 * 193


def test_both_search_loops_are_bounded_by_the_same_step_budget():
    """The walk and the six-offset refinement share one budget, and BOTH have
    to honour it.

    The refinement is the loop that mattered: it used to be `while not found`
    with nothing to stop it, and an agent that respawns clears the memory
    convergence is detected from. Counting in `_step` -- the only place a
    forward pass happens -- is what makes one budget cover both.
    """
    import numpy as np

    from sadt_ali_cbct.agent import Agent, NotFound

    class _OneScaleVolume:
        """Enough environment for the two loops; the walk itself is stubbed."""

        scale_count = 1

        def size(self, _scale):
            return np.array([64, 64, 64])

    # `walk`: never converges, so `search`'s own loop must run out of steps.
    # `refine`: converges ONCE -- enough to leave the walk -- and never again,
    # which is the shape of a respawning agent and is what `_focus` has to
    # survive. It is also the case that used to hang: before the budget
    # reached it, this loop was `while not self._step()` with no way out.
    for phase, converge_once in (("walk", False), ("refine", True)):
        taken = []

        class _Bounded(Agent):
            def _step(self):
                taken.append(1)
                self._steps_left -= 1
                return converge_once and len(taken) == 1

        agent = _Bounded(
            target="X", scale_keys=("sp1",), brain=None,
            environment=_OneScaleVolume(), rng=np.random.default_rng(0),
        )

        with pytest.raises(NotFound) as raised:
            agent.search(40)

        assert "40 steps" in str(raised.value), (phase, str(raised.value))
        assert agent._steps_left <= 0, phase
        # Bounded, not merely finite. `_focus` checks AFTER stepping, so it may
        # overrun by one step per offset; what must not happen is a loop that
        # runs on past the budget indefinitely.
        assert len(taken) <= 40 + 7, (phase, len(taken))


# ---------------------------------------------------------------------------
# A missing dependency belongs to the venv, not to a scan
# ---------------------------------------------------------------------------

def test_a_missing_dependency_fails_before_any_scan_is_touched(tmp_path, monkeypatch):
    """Regression: `itk` absent produced one identical failure PER SCAN -- each
    only after a full histogram correction -- and the run then ended on
    "produced no landmarks for any scan", which buried the one line saying what
    to install. It is a property of the venv; it has to be raised once, before
    the loop."""
    from sadt_ali_cbct import engine as cbct_engine
    from sadt_ali_cbct import preprocess

    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "patient02.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    corrected = []
    monkeypatch.setattr(
        preprocess, "import_itk",
        lambda: (_ for _ in ()).throw(ToolUnavailableError("needs itk (missing: itk)")),
    )
    monkeypatch.setattr(
        preprocess, "correct_histogram",
        lambda *args, **kwargs: corrected.append(args) or args[1],
    )

    with pytest.raises(ToolUnavailableError) as raised:
        cbct_engine.predict_landmarks(
            scans=[(str(tmp_path / "cohort" / "patient01.nii.gz"), "patient01.nii.gz"),
                   (str(tmp_path / "cohort" / "patient02.nii.gz"), "patient02.nii.gz")],
            model_path=bundle,
            regions=("CB",),
            output_dir=str(tmp_path / "out"),
            work_dir=str(tmp_path / "work"),
        )

    # The install message itself, not a summary that hides it.
    assert "missing: itk" in str(raised.value)
    assert "produced no landmarks" not in str(raised.value)
    # And not one scan was preprocessed before finding out.
    assert corrected == []


# ---------------------------------------------------------------------------
# Cross-argument rules the schema cannot express
# ---------------------------------------------------------------------------

def test_an_empty_cbct_selection_on_cbct_input_names_the_argument(tmp_path):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")

    with pytest.raises(ToolInputError) as raised:
        run(
            input=tmp_path / "cohort",
            model=tmp_path,
            output_dir=tmp_path / "out",
            regions=[],
        )
    message = str(raised.value)
    # The argument as PUBLISHED. It read 'cbct_regions' until the split renamed
    # it, so the message was telling a caller to fill a field the schema does
    # not have.
    assert "regions" in message
    # The message lists what to tick, so a mode mismatch explains itself.
    for name in cbct_catalog.REGION_NAMES:
        assert name in message






# ---------------------------------------------------------------------------
# `landmarks` -- asking for named points instead of whole regions
# ---------------------------------------------------------------------------

# What ASO's fully-automated CBCT mode registers on. Straddles two regions,
# which is the whole reason this argument exists.
ASO_LANDMARKS = ("Ba", "S", "N", "RPo", "LPo", "ROr", "LOr")


def _weights_for(labels):
    """A discover_weights() result: every label with both scales present."""
    return {label: {scale: f"/w/{label}/{scale}.pth" for scale in cbct_catalog.SCALE_KEYS}
            for label in labels}


def test_naming_landmarks_replaces_the_region_selection():
    """The point of the argument: 7 agents, not 58.

    ASO's seven points span Cranial base and Upper. Asking by region would run
    every landmark of both (58) to use seven, and one agent is a full two-scale
    walk of the volume.
    """
    from sadt_ali_cbct import engine

    weights = _weights_for(cbct_catalog.LABELS)

    by_region, _missing, _ungrouped = engine.requested_landmarks(
        weights, regions=("CB", "U")
    )
    by_name, missing, _ungrouped = engine.requested_landmarks(
        weights, regions=cbct_catalog.REGION_CODES, landmarks=ASO_LANDMARKS
    )

    assert set(by_name) == set(ASO_LANDMARKS)
    assert not missing
    # The regions were left at their all-on default and still did not widen it.
    assert len(by_region) > 8 * len(by_name)


def test_an_empty_landmark_selection_leaves_the_regions_in_charge():
    """The ordinary case, and the default: an empty list means "not specified"
    and hands the choice back to `cbct_regions`."""
    from sadt_ali_cbct import engine

    weights = _weights_for(cbct_catalog.LABELS)

    without = engine.requested_landmarks(weights, regions=("CB",))
    with_empty = engine.requested_landmarks(weights, regions=("CB",), landmarks=())

    assert without == with_empty
    assert set(without[0]) == set(cbct_catalog.GROUP_LABELS["CB"])


def test_a_named_landmark_the_bundle_lacks_is_reported_not_dropped():
    """Same contract as the region path: "use another bundle" has to be
    distinguishable from "this scan is hard"."""
    from sadt_ali_cbct import engine

    weights = _weights_for(("Ba", "S", "N"))

    runnable, without_model, _ungrouped = engine.requested_landmarks(
        weights, regions=cbct_catalog.REGION_CODES, landmarks=ASO_LANDMARKS
    )

    assert set(runnable) == {"Ba", "S", "N"}
    assert set(without_model) == {"RPo", "LPo", "ROr", "LOr"}


def test_landmark_names_accepts_every_shape_a_caller_uses():
    assert cbct_catalog.landmark_names(["N", "Ba"]) == ("Ba", "N")  # declaration order
    assert cbct_catalog.landmark_names(None) == ()
    assert cbct_catalog.landmark_names([]) == ()
    # Aliases resolve to the spelling the weights are packaged under.
    assert cbct_catalog.landmark_names(["UR3OI"]) == ("UR3OIP",)
    # A name outside the catalog is kept, not refused: the engine runs it if
    # the bundle has weights for it, and reports it as ungrouped otherwise.
    assert cbct_catalog.landmark_names(["Ba", "Mystery"]) == ("Ba", "Mystery")


def test_only_the_named_landmarks_run(tmp_path, stub_agent, cbct_environment):
    """End to end: the seven points ASO needs, from a bundle holding more."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(
        tmp_path / "bundle", {"Cranial_Base": list(ASO_LANDMARKS) + ["C2", "C3"]}
    )

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        landmarks=list(ASO_LANDMARKS),
    )
    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())

    assert set(report["landmarks_requested"]) == set(ASO_LANDMARKS)
    # The regions were left at their all-on default and did not widen it.
    assert "C2" not in report["landmarks_requested"]
    # And the report says the selection came from `landmarks`, not the regions.
    assert report["regions"] == []


# ---------------------------------------------------------------------------
# The real bundle, on the real card
# ---------------------------------------------------------------------------

REAL_CBCT_MODELS = os.environ.get("SADT_ALI_CBCT_MODELS")
REAL_IOS_MODELS = os.environ.get("SADT_ALI_IOS_MODELS")
REAL_SCAN = os.environ.get("SADT_ALI_SCAN")
REAL_MESH = os.environ.get("SADT_ALI_MESH")


@pytest.mark.gpu
@pytest.mark.models
@pytest.mark.skipif(
    not (REAL_CBCT_MODELS and REAL_SCAN),
    reason="set SADT_ALI_CBCT_MODELS and SADT_ALI_SCAN (see tests/data/README.md)",
)
def test_the_real_cbct_bundle_places_landmarks_on_a_real_scan(tmp_path):
    """The shipped bundle on a real CBCT, on the GPU.

    Positions are compared against the pre-port implementation separately (see
    README, "Validated against"); what this asserts is that the real bundle's
    layout, the real scan geometry and the agent's two-scale walk all survive
    the repackaging.

    The seven landmarks are the ones ASO registers on, which is what makes this
    the cheapest run that still covers the path another tool depends on.
    """
    from pathlib import Path

    wanted = ["Ba", "S", "N", "RPo", "LPo", "ROr", "LOr"]
    output = run(
        input=Path(REAL_SCAN),
        model=Path(REAL_CBCT_MODELS),
        output_dir=tmp_path / "out",
        landmarks=wanted,
        device="cuda",
    )

    report = json.loads((output / dispatch.REPORT_NAME).read_text())
    assert report["mode"] == "CBCT"
    assert report["device"].startswith("cuda")
    assert report["summary"] == {"total": 1, "processed": 1, "failed": 0}
    # Named landmarks replace the regions, so exactly these were run.
    assert set(report["landmarks_requested"]) == set(wanted)
    assert report["regions"] == []

    produced = sorted(output.rglob("*.mrk.json"))
    assert len(produced) == 1
    points = json.loads(produced[0].read_text())["markups"][0]["controlPoints"]
    found = {point["label"]: point["position"] for point in points}
    assert set(found) == set(wanted), report["cases"]

    # Inside the scan's own physical extent, which is the cheap check that
    # catches a coordinate convention going wrong -- the failure mode that
    # leaves a perfectly valid file placing every point outside the head.
    import SimpleITK as sitk

    image = sitk.ReadImage(str(REAL_SCAN))
    corners = [
        image.TransformIndexToPhysicalPoint((x, y, z))
        for x in (0, image.GetSize()[0] - 1)
        for y in (0, image.GetSize()[1] - 1)
        for z in (0, image.GetSize()[2] - 1)
    ]
    low = [min(c[axis] for c in corners) for axis in range(3)]
    high = [max(c[axis] for c in corners) for axis in range(3)]
    for label, position in found.items():
        for axis in range(3):
            assert low[axis] <= position[axis] <= high[axis], (label, axis, position)




# ---------------------------------------------------------------------------
# Mucogingival
# ---------------------------------------------------------------------------









def test_a_degraded_landmark_carries_its_caveat_into_the_file(tmp_path):
    """A point placed from an arch fit looks exactly like a good one in the
    file. Whoever opens it is the one who has to know which to review, so the
    caveat travels WITH the point rather than only in the run report."""
    path = markups.write(
        {"LL6MG": (1.0, 2.0, 3.0), "LL5MG": (4.0, 5.0, 6.0)},
        str(tmp_path / "arch_lm_Pred.mrk.json"),
        descriptions={"LL6MG": "position estimated from the arch"},
    )
    points = {
        point["label"]: point["description"]
        for point in json.loads(open(path, encoding="utf-8").read())["markups"][0][
            "controlPoints"
        ]
    }

    assert points["LL6MG"] == "position estimated from the arch"
    # And a point with nothing to say still says nothing.
    assert points["LL5MG"] == ""








def test_an_intraoral_surface_is_refused_by_name(tmp_path):
    """What the split put in place of the `modality` argument.

    ALI used to be one tool that read the mode from the data; five tests here
    drove that through `modality=`, and they went on existing after the split
    because a stale import stopped this file from being collected at all. The
    behaviour they were guarding is now this: the wrong kind of data is refused
    before any inference, and the message names the tool that does handle it.
    """
    root = tmp_path / "scans"
    root.mkdir()
    write_surface(root / "patient.vtk")

    with pytest.raises(ToolInputError) as caught:
        dispatch.discover(str(root))

    message = str(caught.value)
    assert "intraoral surface" in message
    assert "ALI_IOS" in message


# ---------------------------------------------------------------------------
# The seed: the agents' respawn must not come from OS entropy
# ---------------------------------------------------------------------------
#
# The defect these guard: `agent._respawn()` drew from the GLOBAL `np.random`,
# which nothing in this tool ever seeded. An agent that steps out of the volume
# restarts at a random position, so a landmark at the edge of the field of view
# answered differently from one run to the next. Measured on the reference
# scan: six identical requests, three different outputs -- `C4` at one point
# (3/6), at a second 52.99 mm away (1/6), or missing (2/6).


def test_the_respawn_generator_is_fixed_by_the_seed_and_the_landmark():
    from sadt_ali_cbct.agent import rng_for

    first = rng_for("C4", 0).integers(0, 10_000, size=20)
    again = rng_for("C4", 0).integers(0, 10_000, size=20)
    assert np.array_equal(first, again)

    assert not np.array_equal(first, rng_for("C4", 1).integers(0, 10_000, size=20))
    assert not np.array_equal(first, rng_for("C3", 0).integers(0, 10_000, size=20))


def test_the_respawn_generator_is_the_same_in_a_fresh_interpreter():
    """`hash()` on a str is salted per process; `zlib.crc32` is not.

    Seeding from the built-in hash would look correct in one process and hand
    the entropy straight back across two -- which is exactly the failure being
    fixed, only harder to see.
    """
    import subprocess
    import sys

    script = (
        "from sadt_ali_cbct.agent import rng_for; "
        "print(list(rng_for('C4', 0).integers(0, 10_000, size=5)))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": salt},
        ).stdout.strip()
        for salt in ("0", "1", "12345")
    ]
    assert len(set(runs)) == 1, runs


def test_an_agent_cannot_be_built_without_saying_where_its_randomness_comes_from():
    """`rng` is keyword-only and has no default, so the defect cannot come back
    by someone simply forgetting the argument."""
    from sadt_ali_cbct.agent import Agent

    with pytest.raises(TypeError):
        Agent(target="C4", scale_keys=("1", "0-3"), brain=None, environment=None)


def test_two_agents_with_the_same_seed_respawn_to_the_same_place():
    from sadt_ali_cbct.agent import Agent, rng_for

    class FlatEnvironment:
        """Just enough environment for `_respawn`: the volume's shape."""

        scale_count = 2

        def size(self, _scale_key):
            return np.array([120, 130, 140])

    def respawns(seed, count=5):
        agent = Agent(
            target="C4", scale_keys=("1", "0-3"), brain=None,
            environment=FlatEnvironment(), rng=rng_for("C4", seed),
        )
        positions = []
        for _ in range(count):
            agent._respawn()
            positions.append(tuple(int(value) for value in agent.position))
        return positions

    assert respawns(0) == respawns(0)
    assert respawns(0) != respawns(7)


def test_respawning_does_not_touch_the_global_numpy_stream():
    """The other half of the same defect, and the reason a LOCAL generator is
    used rather than a global seed: a shared stream means two concurrent runs
    in one process consume each other's randomness."""
    from sadt_ali_cbct.agent import Agent, rng_for

    class FlatEnvironment:
        scale_count = 2

        def size(self, _scale_key):
            return np.array([120, 130, 140])

    agent = Agent(
        target="C4", scale_keys=("1", "0-3"), brain=None,
        environment=FlatEnvironment(), rng=rng_for("C4", 0),
    )

    np.random.seed(1234)
    expected = np.random.random(4)

    np.random.seed(1234)
    for _ in range(10):
        agent._respawn()
    assert np.array_equal(np.random.random(4), expected)


def test_the_run_report_records_the_seed(tmp_path, stub_agent, cbct_environment):
    """A markups file plus this line is enough to ask for the same answer
    again, without knowing how the request was made."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        regions=CRANIAL_BASE_ONLY,
        seed=4242,
    )

    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())
    assert report["seed"] == 4242


def test_a_seed_outside_the_accepted_range_is_an_input_error(
    tmp_path, stub_agent, cbct_environment
):
    """Refused with a message the client shows verbatim, rather than a numpy
    traceback from four frames down."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    for bad in (-1, 2 ** 32):
        with pytest.raises(ToolInputError, match="seed must be between"):
            run(
                input=tmp_path / "cohort",
                model=bundle,
                output_dir=tmp_path / "out",
                regions=CRANIAL_BASE_ONLY,
                seed=bad,
            )


def test_a_batch_says_which_scan_it_is_on(tmp_path, stub_agent, cbct_environment, monkeypatch):
    """One event per scan, rising, and the position rather than the name.

    Both scans here are called `patient01.nii.gz` -- the very case the output
    tree exists to keep apart -- so the name would identify neither the patient
    nor the position, and it is patient metadata besides. The log line beside
    this call has said so since the port; the progress message obeys the same
    rule because it is stored on the server and shown to a watcher.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    write_volume(tmp_path / "cohort" / "siteA" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "siteB" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        regions=CRANIAL_BASE_ONLY,
    )

    events = [json.loads(line) for line in events_file.read_text().splitlines() if line]
    assert [event["message"] for event in events] == ["scan 1 of 2", "scan 2 of 2"]
    assert [event["fraction"] for event in events] == [0.0, 0.5]
    assert not any("patient01" in event["message"] for event in events)


def test_every_description_is_sourced_or_absent():
    """A landmark with no line shows no tooltip, which is what it had before.
    A landmark with a WRONG line is a point placed wrongly, so the table covers
    only what Gillot et al. 2023 Table 1 reaches."""
    from sadt_ali_cbct import catalog

    offered = {label for labels in catalog.GROUP_LABELS.values() for label in labels}
    assert set(catalog.DESCRIPTIONS) <= offered
    assert all(text.strip() for text in catalog.DESCRIPTIONS.values())
    # The three suffixes the paper does not define stay undescribed.
    assert not [label for label in catalog.DESCRIPTIONS
                if label.endswith(("MP", "OIP", "RIP"))]


def test_a_suffix_reads_differently_along_the_arch():
    """`O` is an incisal edge on an incisor and a cusp tip from the canine
    back; `R` is a root canal on anything anterior and a pulp chamber floor on
    a molar. The two do not change at the same tooth."""
    from sadt_ali_cbct import catalog

    assert "incisal edge" in catalog.DESCRIPTIONS["UR1O"]
    assert "cusp tip" in catalog.DESCRIPTIONS["UL3O"]
    assert "root canal" in catalog.DESCRIPTIONS["UR3R"]
    assert "pulp chamber" in catalog.DESCRIPTIONS["UR6R"]


def test_the_words_travel_with_the_argument():
    from sadt_ali_cbct import catalog, layout

    assert layout.LAYOUT["landmarks"]["option_help"] is catalog.DESCRIPTIONS


def test_the_frankfort_horizontal_points_are_named():
    """ASO registers on `Ba, S, N, RPo, LPo, ROr, LOr` and publishes that
    reference as "Frankfurt Horizontal + Midsagittal" -- a plane defined by
    porion and orbitale. The reference's own name is what identifies them, so
    the seven landmarks a clinician meets most often all carry a line."""
    from sadt_ali_cbct import catalog

    for label in ("Ba", "S", "N", "RPo", "LPo", "ROr", "LOr"):
        assert label in catalog.DESCRIPTIONS, label
    assert "porion" in catalog.DESCRIPTIONS["RPo"].lower()
    assert "orbitale" in catalog.DESCRIPTIONS["LOr"].lower()


# ---------------------------------------------------------------------------
# How many agents walk at once
# ---------------------------------------------------------------------------

def test_the_ask_is_capped_at_the_widest_width_worth_having():
    """A run RESERVES what it is granted, so asking wider than the measured
    knee takes room from other runs and gives this one nothing.

    Measured on 119 landmarks: width 8 is 128.0 s and width 12 is 127.3 s, for
    three more gigabytes of card. The cap is on the ASK -- the supervisor still
    answers with less on a busy machine, and a request holding fewer landmarks
    than the cap still asks only for those.
    """
    from sadt_ali_cbct import engine as cbct_engine

    asked = []

    class _Generous:
        """A supervisor that grants whatever it is asked for."""

        def channels(self, wanted):
            asked.append(wanted)
            return wanted

    # 119 landmarks on a machine that would pay for all of them: still 8.
    assert cbct_engine._channels_for(_Generous(), 119) == cbct_engine.MAX_AGENT_CHANNELS
    assert asked == [cbct_engine.MAX_AGENT_CHANNELS], (
        "the cap must narrow the ASK, not just the answer -- admission reserves "
        "against what the tool asked for"
    )

    # Fewer landmarks than the cap: the count still decides.
    assert cbct_engine._channels_for(_Generous(), 3) == 3

    class _Stingy:
        def channels(self, wanted):
            return 2

    # A busy machine still narrows it further; the cap is not a floor.
    assert cbct_engine._channels_for(_Stingy(), 119) == 2

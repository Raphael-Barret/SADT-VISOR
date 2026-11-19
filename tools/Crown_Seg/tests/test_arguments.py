"""Every argument, at every value it can take, including the boundaries.

`run()`'s signature is the schema the client renders, so each of these is a
widget a clinician can move. What a value does has to be pinned where it is
decided, not inferred from a happy-path run.
"""

import json
import os

import pytest

from sadt_crownseg import pipeline, run
from sadt_crownseg.errors import ToolInputError

from conftest import write_stl, write_surface


def read_report(output_dir):
    with open(os.path.join(str(output_dir), "run_report.json"), encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# array_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "given, expected",
    [
        (None, pipeline.DEFAULT_ARRAY_NAME),
        ("", pipeline.DEFAULT_ARRAY_NAME),
        ("   ", pipeline.DEFAULT_ARRAY_NAME),
        ("  PredictedID  ", "PredictedID"),
        ("Universal_ID", "Universal_ID"),
    ],
)
def test_array_name_falls_back_to_the_default_when_it_says_nothing(
    tmp_path, stub_engine, model_file, given, expected
):
    """An empty text field arrives as an empty string, not as an omitted
    argument. Writing the labels into an array called `""` would produce a mesh
    no downstream tool can read."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
        array_name=given,
    )

    assert report["array_name"] == expected
    assert stub_engine[0]["array_name"] == expected


def test_a_pass_through_keeps_the_array_it_already_had(tmp_path, stub_engine, model_file):
    """`array_name` names where the NETWORK writes its labels. A mesh that was
    not re-predicted keeps whatever array it arrived with -- renaming it would
    claim this run produced labels it only copied."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True, array_name="Universal_ID")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
        array_name="PredictedID",
    )

    import vtk

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(report["segmented_meshes"][0])
    reader.Update()
    point_data = reader.GetOutput().GetPointData()
    names = {point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())}
    assert names == {"Universal_ID"}
    assert pipeline.is_segmented(report["segmented_meshes"][0])


# ---------------------------------------------------------------------------
# suffix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "given, expected",
    [
        (None, pipeline.DEFAULT_SUFFIX),
        ("", pipeline.DEFAULT_SUFFIX),
        ("  ", pipeline.DEFAULT_SUFFIX),
        ("  Pred  ", "Pred"),
        ("CrownSeg", "CrownSeg"),
    ],
)
def test_suffix_falls_back_to_the_default_when_it_says_nothing(
    tmp_path, stub_engine, model_file, given, expected
):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
        suffix=given,
    )

    assert report["suffix"] == expected
    assert os.path.basename(report["segmented_meshes"][0]) == f"arch_{expected}.vtk"


def test_the_suffix_names_the_pass_through_output_too(tmp_path, stub_engine, model_file):
    """Both branches produce `<base>_<suffix>.vtk`, so a caller cannot tell
    from a file name which one a mesh took."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
        suffix="Pred",
    )

    assert os.path.basename(report["segmented_meshes"][0]) == "arch_Pred.vtk"


# ---------------------------------------------------------------------------
# numbering / fdi
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "numbering, fdi, recorded",
    [("Universal", 0, "Universal"), ("FDI", 1, "FDI")],
)
def test_numbering_reaches_the_engine_as_the_flag_it_expects(
    tmp_path, stub_engine, model_file, numbering, fdi, recorded
):
    """It changes the integers written into the array, not the mesh, and
    whatever consumes the result has to agree -- so the report records it."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    output = run(
        meshes=tmp_path / "cohort",
        model=model_file,
        output_dir=tmp_path / "out",
        numbering=numbering,
    )

    assert stub_engine[0]["fdi"] is bool(fdi)
    assert read_report(output)["numbering"] == recorded


# ---------------------------------------------------------------------------
# skip_segmented
# ---------------------------------------------------------------------------

def test_skip_segmented_true_is_the_default(tmp_path, stub_engine, model_file):
    """What makes a mixed batch of raw and pre-segmented meshes one call."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert stub_engine == []
    assert report["summary"]["already_segmented"] == 1


def test_skip_segmented_false_sends_a_labelled_mesh_to_the_engine(
    tmp_path, stub_engine, model_file
):
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
        skip_segmented=False,
    )

    assert len(stub_engine) == 1
    assert report["summary"] == {
        "total": 1, "segmented": 1, "already_segmented": 0, "failed": 0,
        "engine_unavailable": 0,
    }


def test_a_mixed_batch_splits_between_the_two_branches(tmp_path, stub_engine, model_file):
    write_surface(tmp_path / "cohort" / "raw.vtk")
    write_surface(tmp_path / "cohort" / "done.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["summary"]["segmented"] == 1
    assert report["summary"]["already_segmented"] == 1
    assert report["cases"]["raw.vtk"]["status"] == "segmented"
    assert report["cases"]["done.vtk"]["status"] == "already_segmented"


def test_only_the_raw_meshes_are_listed_in_the_engine_csv(tmp_path, stub_engine, model_file):
    """Sending a labelled mesh to the network anyway would cost minutes per
    mesh for an array that is already there."""
    write_surface(tmp_path / "cohort" / "raw.vtk")
    write_surface(tmp_path / "cohort" / "done.vtk", labelled=True)

    pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    # The csv is gone with the work dir; the stub recorded what it held.
    assert len(stub_engine) == 1


def test_an_stl_always_goes_to_the_engine(tmp_path, stub_engine, model_file):
    """STL carries no point data by construction, so `skip_segmented` can never
    apply to one."""
    write_stl(tmp_path / "cohort" / "arch.stl", tmp_path=tmp_path)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert len(stub_engine) == 1
    assert report["summary"]["segmented"] == 1
    assert report["segmented_meshes"][0].endswith("arch_Seg.vtk")


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requested", [None, "", "cpu", "CPU", " cpu "])
def test_a_cpu_request_resolves_to_cpu(requested):
    assert pipeline.resolve_device(requested) == "cpu"


@pytest.mark.parametrize("requested", ["cuda", "CUDA", "cuda:1", " cuda "])
def test_a_cuda_request_resolves_to_the_first_card(monkeypatch, requested):
    """Always `cuda:0`: a tool is one process with whatever
    `CUDA_VISIBLE_DEVICES` gave it, so the index it names is the server's, not
    the machine's."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert pipeline.resolve_device(requested) == "cuda:0"


def test_cuda_falls_back_to_cpu_with_a_warning_when_no_card_is_visible(monkeypatch, caplog):
    """A CPU deployment must run the tool rather than fail it -- but silently
    running a 52-second job on the CPU for an hour is the other failure."""
    import logging

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with caplog.at_level(logging.WARNING):
        assert pipeline.resolve_device("cuda") == "cpu"

    assert "CUDA is unavailable" in caplog.text


@pytest.mark.parametrize("requested", ["mps", "MPS", " mps "])
def test_an_mps_request_is_honoured_on_a_mac_with_metal(monkeypatch, requested):
    """It used to resolve to the CPU, because shapeaxi accepted nothing else:
    the Apple GPU was never reached from this tool. The engine has no such
    limit, so the request now means what it says."""
    monkeypatch.setattr(pipeline.sys, "platform", "darwin")
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert pipeline.resolve_device(requested) == "mps"


def test_mps_falls_back_to_cpu_with_a_warning_when_metal_is_unavailable(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(pipeline.sys, "platform", "darwin")
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    with caplog.at_level(logging.WARNING):
        assert pipeline.resolve_device("mps") == "cpu"

    assert "MPS is unavailable" in caplog.text


def test_mps_is_never_used_off_a_mac(monkeypatch, caplog):
    """A Linux build of torch answers `torch.backends.mps.is_available()` with
    False anyway; the platform is checked as well so the answer does not depend
    on a build."""
    import logging

    monkeypatch.setattr(pipeline.sys, "platform", "linux")
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    with caplog.at_level(logging.WARNING):
        assert pipeline.resolve_device("mps") == "cpu"

    assert "MPS is unavailable" in caplog.text


def test_the_resolved_device_is_what_reaches_the_engine_and_the_report(
    tmp_path, monkeypatch, model_file
):
    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cuda:0")
    calls = []
    monkeypatch.setattr(
        pipeline, "_run_engine",
        lambda **kwargs: calls.append(kwargs["device"]) or write_surface(
            pipeline._predicted_path(
                kwargs["output_dir"], "crownseg_input", kwargs["suffix"],
                str(tmp_path / "cohort" / "arch.vtk"), kwargs["input_root"],
            ),
            labelled=True,
        ),
    )
    write_surface(tmp_path / "cohort" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
        device="cuda",
    )

    assert calls == ["cuda:0"]
    assert report["device"] == "cuda:0"


def test_a_run_that_segmented_nothing_records_no_device(tmp_path, stub_engine, model_file):
    """A pure pass-through never resolved one, and reporting "cpu" would claim
    a run that did not happen."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["device"] is None


# ---------------------------------------------------------------------------
# num_workers
# ---------------------------------------------------------------------------

def test_num_workers_travels_from_run_to_the_engine(tmp_path, stub_engine, model_file):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    run(
        meshes=tmp_path / "cohort",
        model=model_file,
        output_dir=tmp_path / "out",
        num_workers=7,
    )

    assert stub_engine[0]["num_workers"] == 7


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def test_a_missing_checkpoint_names_the_path_it_looked_for(tmp_path, stub_engine):
    """The model is an argument now, not a name looked up in a data store, so
    the message has to say which file was expected."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(ToolInputError) as failure:
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(tmp_path / "absent.pth"),
            output_dir=str(tmp_path / "out"),
        )

    assert str(tmp_path / "absent.pth") in str(failure.value)


def test_a_directory_with_no_checkpoint_in_it_is_refused(tmp_path, stub_engine):
    """A FOLDER is now a legitimate value: the server hands this tool its whole
    models directory, and a supervised caller passes on what it was given.

    So the refusal moved rather than disappearing -- what is refused is a folder
    with no crown checkpoint in it, and the message says what it looked for. An
    empty folder still must not reach the engine.
    """
    write_surface(tmp_path / "cohort" / "arch.vtk")
    (tmp_path / "models").mkdir()

    with pytest.raises(ToolInputError, match="holds no crown-segmentation checkpoint"):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(tmp_path / "models"),
            output_dir=str(tmp_path / "out"),
        )


def test_the_checkpoint_path_is_what_reaches_the_engine(tmp_path, stub_engine, model_file):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert stub_engine[0]["model"] == str(model_file)


def test_run_accepts_paths_and_returns_one(tmp_path, stub_engine, model_file):
    """`run()` is what the runner calls with values out of `job.json`."""
    from pathlib import Path

    write_surface(tmp_path / "cohort" / "arch.vtk")

    output = run(meshes=tmp_path / "cohort", model=model_file, output_dir=tmp_path / "out")

    assert isinstance(output, Path)
    assert output == tmp_path / "out"

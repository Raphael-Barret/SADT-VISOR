"""The output tree, the run report, and what must never be in either.

The report is the tool's only structured answer: the server sequences CrownSeg
before ALI by reading `segmented_meshes` out of it, and it travels to a client
inside the returned archive.
"""

import json
import logging
import os
from pathlib import Path

import pytest

from sadt_crownseg import pipeline, run
from sadt_crownseg.errors import ToolInputError, ToolUnavailableError

from conftest import write_stl, write_surface


def read_report(output_dir):
    with open(Path(output_dir) / "run_report.json", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# The output directory
# ---------------------------------------------------------------------------

def test_the_output_directory_is_created_when_it_does_not_exist(
    tmp_path, stub_engine, model_file
):
    destination = tmp_path / "does" / "not" / "exist"
    write_surface(tmp_path / "cohort" / "arch.vtk")

    run(meshes=tmp_path / "cohort", model=model_file, output_dir=destination)

    assert destination.is_dir()


def test_a_refused_request_creates_no_output_directory_at_all(tmp_path, stub_engine):
    """A 422 that has already written a scratch folder into the response's own
    directory is a refusal that made a mess. Both refusals are checked: the
    checkpoint and the input."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(ToolInputError):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(tmp_path / "absent.pth"),
            output_dir=str(tmp_path / "out"),
        )
    assert not (tmp_path / "out").exists()

    (tmp_path / "empty").mkdir()
    with pytest.raises(ToolInputError):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "empty"),
            model_path=str(tmp_path / "model.pth"),
            output_dir=str(tmp_path / "out"),
        )
    assert not (tmp_path / "out").exists()


def test_the_work_directory_does_not_survive_a_successful_run(
    tmp_path, stub_engine, model_file
):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    output = run(meshes=tmp_path / "cohort", model=model_file, output_dir=tmp_path / "out")

    assert not (output / pipeline.WORK_DIRNAME).exists()


def test_the_work_directory_does_not_survive_an_engine_that_raised(
    tmp_path, monkeypatch, model_file
):
    """It holds a csv listing the patient's own file paths, so it must go
    whichever way the run ends."""
    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")

    def explode(**kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "_run_engine", explode)
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(model_file),
            output_dir=str(tmp_path / "out"),
        )

    assert not (tmp_path / "out" / pipeline.WORK_DIRNAME).exists()


def test_the_work_directory_does_not_survive_a_run_that_produced_nothing(
    tmp_path, failing_engine, model_file
):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(RuntimeError, match="0 of 1 meshes segmented; the engine returned without error"):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(model_file),
            output_dir=str(tmp_path / "out"),
        )

    assert not (tmp_path / "out" / pipeline.WORK_DIRNAME).exists()


def test_the_input_csv_is_written_under_the_output_directory(
    tmp_path, stub_engine, model_file
):
    """The Slicer module wrote it into the extension's own source folder, which
    breaks a read-only install and leaves a file behind pointing at the
    patient's data."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    csv_path = stub_engine[0]["csv"]
    assert csv_path.startswith(str(tmp_path / "out"))
    assert pipeline.WORK_DIRNAME in csv_path
    assert not os.path.exists(csv_path)


def test_nothing_is_written_outside_the_output_directory(tmp_path, stub_engine, model_file):
    write_surface(tmp_path / "cohort" / "arch.vtk")
    before = sorted(path for path in tmp_path.rglob("*") if path.is_file())

    run(meshes=tmp_path / "cohort", model=model_file, output_dir=tmp_path / "out")

    after = sorted(
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(tmp_path / "out")
    )
    assert after == before


def test_the_input_meshes_are_left_exactly_as_they_were(tmp_path, stub_engine, model_file):
    """The input may live on a read-only mount, and a labelled mesh is COPIED
    rather than referenced in place for that reason."""
    mesh = Path(write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True))
    before = mesh.read_bytes()

    run(meshes=tmp_path / "cohort", model=model_file, output_dir=tmp_path / "out")

    assert mesh.read_bytes() == before


# ---------------------------------------------------------------------------
# The output tree
# ---------------------------------------------------------------------------

def test_a_batch_keeps_its_tree_so_two_patients_cannot_collide(
    tmp_path, stub_engine, model_file
):
    """Flattened, one `arch.vtk` would overwrite the other and the caller would
    silently get half its cohort back."""
    write_surface(tmp_path / "cohort" / "siteA" / "arch.vtk")
    write_surface(tmp_path / "cohort" / "siteB" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert len(set(report["segmented_meshes"])) == 2
    assert sorted(report["cases"]) == [
        os.path.join("siteA", "arch.vtk"), os.path.join("siteB", "arch.vtk")
    ]


def test_a_pass_through_mirrors_the_input_tree_directly_under_the_output(
    tmp_path, stub_engine, model_file
):
    write_surface(tmp_path / "cohort" / "siteA" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["segmented_meshes"] == [str(tmp_path / "out" / "siteA" / "arch_Seg.vtk")]


def test_a_segmented_mesh_lands_under_shapeaxis_own_folder(tmp_path, stub_engine, model_file):
    """shapeaxi's csv branch files its outputs under `<csv stem>_<suffix>/`,
    and the tool recomputes that path rather than parsing it back out of
    stdout -- which is where the patient's file names would have travelled. The
    two branches therefore produce two trees, which is what the README says."""
    write_surface(tmp_path / "cohort" / "siteA" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["segmented_meshes"] == [
        str(tmp_path / "out" / "crownseg_input_Seg" / "siteA" / "arch_Seg.vtk")
    ]


def test_the_predicted_path_follows_the_suffix_into_the_folder_name(tmp_path):
    """Both halves of shapeaxi's layout use the suffix: the folder and the
    file. A suffix that reached one and not the other would make the tool look
    for its own outputs in the wrong place and report every mesh failed."""
    predicted = pipeline._predicted_path(
        output_dir="/out", csv_stem="crownseg_input", suffix="Pred",
        mesh="/in/cohort/siteA/arch.vtk", input_root="/in/cohort",
    )

    assert predicted == os.path.join(
        "/out", "crownseg_input_Pred", "siteA", "arch_Pred.vtk"
    )


def test_every_output_is_vtk_whichever_branch_a_mesh_took(tmp_path, stub_engine, model_file):
    """A pre-segmented .stl is impossible, but a raw one is routine, and the
    caller gets one format either way instead of an .stl every downstream
    reader has to special-case."""
    write_stl(tmp_path / "cohort" / "raw.stl", tmp_path=tmp_path)
    write_surface(tmp_path / "cohort" / "done.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert len(report["segmented_meshes"]) == 2
    assert all(path.endswith(".vtk") for path in report["segmented_meshes"])


def test_a_single_mesh_lands_at_the_top_of_the_output(tmp_path, stub_engine, model_file):
    """Not under a reconstruction of its absolute path on the server."""
    mesh = write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=mesh,
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["segmented_meshes"] == [str(tmp_path / "out" / "arch_Seg.vtk")]
    assert list(report["cases"]) == ["arch.vtk"]


def test_writing_a_mesh_as_vtk_reads_an_stl_with_the_right_reader(tmp_path):
    """The one conversion `_write_as_vtk` exists for. Reading an .stl with the
    polydata reader does not fail -- it yields an EMPTY mesh, and the caller
    gets a .vtk with no geometry in it, which is why the point count is what is
    asserted rather than the file's existence."""
    import vtk

    source = write_stl(tmp_path / "arch.stl", tmp_path=tmp_path)

    pipeline._write_as_vtk(source, str(tmp_path / "out.vtk"))

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(tmp_path / "out.vtk"))
    reader.Update()
    assert reader.GetOutput().GetNumberOfPoints() == 3
    assert reader.GetOutput().GetNumberOfCells() == 1
    assert pipeline.is_segmented(str(tmp_path / "out.vtk")) is False


def test_a_pass_through_keeps_the_geometry_it_was_given(tmp_path, stub_engine, model_file):
    """The bypass copies rather than re-predicts, so the mesh a caller gets
    back has to be the mesh it sent."""
    import vtk

    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(report["segmented_meshes"][0])
    reader.Update()
    assert reader.GetOutput().GetNumberOfPoints() == 3
    assert reader.GetOutput().GetNumberOfCells() == 1


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_is_written_beside_the_results_and_is_what_run_returns(
    tmp_path, stub_engine, model_file
):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    returned = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert read_report(tmp_path / "out") == returned


def test_the_report_names_the_tool_the_schema_names(tmp_path, stub_engine, model_file):
    """Derived from the folder, not written down: this file said "CrownSeg"
    long after the folder was renamed, and nothing noticed."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["tool"] == pipeline.TOOL_NAME == "Crown_Seg"


def test_the_report_carries_every_field_a_caller_reads(tmp_path, stub_engine, model_file):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert set(report) == {
        "tool", "array_name", "suffix", "numbering", "device", "cases",
        "segmented_meshes", "engine_available", "engine_error", "summary",
        "duration_seconds",
        # WHICH weights ran, by name: `model` may be a folder holding several,
        # so the argument no longer answers that on its own.
        "checkpoint",
    }
    assert set(report["summary"]) == {
        "total", "segmented", "already_segmented", "engine_unavailable", "failed"
    }
    assert isinstance(report["duration_seconds"], float)


def test_the_summary_counts_agree_with_the_per_mesh_statuses(
    tmp_path, stub_engine, model_file
):
    """Counted off the records rather than off the split, so a degraded run
    cannot report N already_segmented AND N engine_unavailable."""
    write_surface(tmp_path / "cohort" / "raw.vtk")
    write_surface(tmp_path / "cohort" / "done.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    statuses = [record["status"] for record in report["cases"].values()]
    summary = report["summary"]
    assert summary["total"] == len(statuses)
    for status in ("segmented", "already_segmented", "engine_unavailable", "failed"):
        assert summary[status] == statuses.count(status), status


def test_the_report_lists_every_mesh_that_now_carries_labels(
    tmp_path, stub_engine, model_file
):
    """The server runs Crown_Seg then ALI, and this list is what it reads --
    ALI no longer imports this tool to find out."""
    write_surface(tmp_path / "cohort" / "raw.vtk")
    write_surface(tmp_path / "cohort" / "done.vtk", labelled=True)

    output = run(meshes=tmp_path / "cohort", model=model_file, output_dir=tmp_path / "out")
    report = read_report(output)

    assert len(report["segmented_meshes"]) == 2
    assert all(pipeline.is_segmented(path) for path in report["segmented_meshes"])
    assert all(Path(path).is_relative_to(output) for path in report["segmented_meshes"])
    assert all(Path(path).is_file() for path in report["segmented_meshes"])


def test_the_report_names_no_path_outside_the_output_directory(
    tmp_path, stub_engine, model_file
):
    """It travels to a client inside the archive. The input folder, the
    checkpoint and the scratch directory are all server-side, and none of them
    belongs in it."""
    write_surface(tmp_path / "cohort" / "raw.vtk")
    write_surface(tmp_path / "cohort" / "done.vtk", labelled=True)

    output = run(meshes=tmp_path / "cohort", model=model_file, output_dir=tmp_path / "out")
    text = (output / "run_report.json").read_text(encoding="utf-8")

    assert str(tmp_path / "cohort") not in text
    assert str(model_file) not in text
    assert pipeline.WORK_DIRNAME not in text


def test_a_produced_mesh_carries_no_trace_of_where_it_came_from(
    tmp_path, stub_engine, model_file
):
    """The .vtk itself travels too, and a VTK header can carry a comment."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    output = run(meshes=tmp_path / "cohort", model=model_file, output_dir=tmp_path / "out")

    produced = (output / "arch_Seg.vtk").read_bytes()
    assert str(tmp_path / "cohort").encode() not in produced


# ---------------------------------------------------------------------------
# A mesh that failed, and a run that failed
# ---------------------------------------------------------------------------

def test_one_mesh_the_engine_could_not_write_does_not_cost_the_batch(
    tmp_path, monkeypatch, model_file
):
    """Forty patients, one unreadable file: the other thirty-nine must come
    back."""
    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")

    def only_the_first(csv_path, output_dir, model_path, input_root, array_name, suffix,
                      device, fdi, num_workers=2):
        with open(csv_path, encoding="utf-8") as handle:
            meshes = [line.strip() for line in handle.read().splitlines()[1:] if line.strip()]
        write_surface(
            pipeline._predicted_path(output_dir, "crownseg_input", suffix, meshes[0],
                                     input_root),
            labelled=True,
        )

    monkeypatch.setattr(pipeline, "_run_engine", only_the_first)
    write_surface(tmp_path / "cohort" / "a.vtk")
    write_surface(tmp_path / "cohort" / "b.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["summary"] == {
        "total": 2, "segmented": 1, "already_segmented": 0, "failed": 1,
        "engine_unavailable": 0,
    }
    assert report["cases"]["b.vtk"]["status"] == "failed"
    assert "no output" in report["cases"]["b.vtk"]["error"]
    assert "output" not in report["cases"]["b.vtk"]


def test_a_run_that_produced_nothing_at_all_is_a_failure_not_an_empty_report(
    tmp_path, failing_engine, model_file
):
    """An empty archive and a green status is the worst answer available."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(RuntimeError, match="0 of 1 meshes segmented; the engine returned without error"):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(model_file),
            output_dir=str(tmp_path / "out"),
        )

    assert not (tmp_path / "out" / "run_report.json").exists()


# ---------------------------------------------------------------------------
# A deployment that could not have segmented anything
# ---------------------------------------------------------------------------

def test_a_pass_through_batch_is_still_served_when_the_engine_is_absent(
    tmp_path, absent_engine, model_file
):
    """Refusing it would break the Crown_Seg -> ALI chain for callers who
    segmented elsewhere. It is recorded instead."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["summary"]["engine_unavailable"] == 1
    assert report["summary"]["already_segmented"] == 0
    assert report["cases"]["arch.vtk"]["status"] == "engine_unavailable"
    assert len(report["segmented_meshes"]) == 1


def test_the_report_says_why_the_engine_was_unavailable(tmp_path, absent_engine, model_file):
    """Without it, "nothing needed segmenting" and "nothing COULD have been
    segmented" produce an identical report."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["engine_available"] is False
    assert report["engine_error"].startswith("ToolUnavailableError: ")
    assert "uv sync --extra segmentation" in report["engine_error"]


def test_a_working_engine_reports_itself_available(tmp_path, stub_engine, model_file):
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(model_file),
        output_dir=str(tmp_path / "out"),
    )

    assert report["engine_available"] is True
    assert report["engine_error"] is None
    assert report["cases"]["arch.vtk"]["status"] == "already_segmented"


def test_a_raw_mesh_with_no_engine_fails_the_run_rather_than_reporting_success(
    tmp_path, absent_engine, model_file
):
    """The probe is recorded, not raised -- but a mesh that actually needs the
    network still reaches it, and gets the 501 the server answers with."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(ToolUnavailableError, match="uv sync --extra segmentation"):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(model_file),
            output_dir=str(tmp_path / "out"),
        )


# ---------------------------------------------------------------------------
# What an operator reads when a run fails
# ---------------------------------------------------------------------------

DATALOADER_FAILURE = (
    "Caught ValueError in DataLoader worker process 0.\n"
    "Original Traceback (most recent call last):\n"
    '  File "/venv/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", '
    "line 308, in _worker_loop\n"
    "    data = fetcher.fetch(index)\n"
    '  File "/venv/lib/python3.10/site-packages/shapeaxi/saxi_dataset.py", line 70, '
    "in __getitem__\n"
    "    raise ValueError(msg)\n"
    "ValueError: mesh has 0 points\n"
)


def test_the_cause_of_a_dataloader_failure_is_its_last_line():
    """The wrapper and the frame list come first and say nothing; the server
    keeps only the head of a reason, so the cause must be pulled out."""
    assert pipeline._last_meaningful_line(DATALOADER_FAILURE) == "ValueError: mesh has 0 points"
    assert pipeline._describe_failure(RuntimeError(DATALOADER_FAILURE)) == (
        "ValueError: mesh has 0 points"
    )


def test_a_one_line_failure_keeps_its_own_type():
    assert pipeline._describe_failure(RuntimeError("CUDA out of memory")) == (
        "RuntimeError: CUDA out of memory"
    )
    assert pipeline._describe_failure(RuntimeError("")) == "RuntimeError"


def test_a_very_long_cause_is_cut_to_fit_the_reason():
    cause = pipeline._last_meaningful_line("x" * 1000)
    assert len(cause) == pipeline.MAX_CAUSE
    assert cause.endswith("...")


def _engine_failing_after_the_first(monkeypatch, message):
    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")

    def write_one_then_fail(csv_path, output_dir, model_path, input_root, array_name,
                            suffix, device, fdi, num_workers=2):
        with open(csv_path, encoding="utf-8") as handle:
            meshes = [line.strip() for line in handle.read().splitlines()[1:] if line.strip()]
        write_surface(
            pipeline._predicted_path(output_dir, "crownseg_input", suffix, meshes[0],
                                     input_root),
            labelled=True,
        )
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "_run_engine", write_one_then_fail)


def test_an_engine_failure_says_what_failed_and_how_far_it_got(
    tmp_path, monkeypatch, model_file, caplog
):
    _engine_failing_after_the_first(monkeypatch, DATALOADER_FAILURE)
    for name in ("a", "b", "c"):
        write_surface(tmp_path / "cohort" / f"{name}.vtk")

    with caplog.at_level(logging.INFO, logger="sadt_crownseg"):
        with pytest.raises(RuntimeError) as failure:
            pipeline.segment_crowns(
                input_path=str(tmp_path / "cohort"),
                model_path=str(model_file),
                output_dir=str(tmp_path / "out"),
            )

    message = str(failure.value)
    assert message == (
        "crown segmentation failed: ValueError: mesh has 0 points "
        "(1 of 3 meshes were written before it)"
    )
    assert type(failure.value) is RuntimeError
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert "Original Traceback" not in message
    assert "segmenting 3 mesh(es) on cpu" in caplog.text
    assert (
        "the engine failed after 1 of 3 meshes were written "
        "(ValueError: mesh has 0 points)"
    ) in caplog.text
    assert "worker.py" not in caplog.text


def test_a_value_error_out_of_the_engine_is_not_answered_as_the_callers_fault(
    tmp_path, monkeypatch, model_file
):
    """ValueError maps to 422; a crash inside the bundled engine is a 500."""
    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")

    def explode(**kwargs):
        raise ValueError("size mismatch for model.0.conv.unit0.conv.weight")

    monkeypatch.setattr(pipeline, "_run_engine", explode)
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(RuntimeError) as failure:
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(model_file),
            output_dir=str(tmp_path / "out"),
        )

    assert not isinstance(failure.value, ValueError)
    assert "ValueError: size mismatch" in str(failure.value)
    assert "0 of 1 meshes were written" in str(failure.value)


def test_an_engine_that_writes_nothing_says_it_returned_without_error(
    tmp_path, failing_engine, model_file, caplog
):
    for name in ("a", "b"):
        write_surface(tmp_path / "cohort" / f"{name}.vtk")

    with caplog.at_level(logging.WARNING, logger="sadt_crownseg"):
        with pytest.raises(RuntimeError) as failure:
            pipeline.segment_crowns(
                input_path=str(tmp_path / "cohort"),
                model_path=str(model_file),
                output_dir=str(tmp_path / "out"),
            )

    assert str(failure.value) == (
        "0 of 2 meshes segmented; the engine returned without error but wrote "
        "none of the 2 expected outputs"
    )
    assert "mesh 1 of 2: segmentation failed" in caplog.text
    assert "mesh 2 of 2: segmentation failed" in caplog.text


def test_a_complete_run_closes_with_an_info_summary(
    tmp_path, stub_engine, model_file, caplog
):
    write_surface(tmp_path / "cohort" / "raw.vtk")
    write_surface(tmp_path / "cohort" / "done.vtk", labelled=True)

    with caplog.at_level(logging.INFO, logger="sadt_crownseg"):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(model_file),
            output_dir=str(tmp_path / "out"),
        )

    summary = [r for r in caplog.records if "meshes labelled" in r.getMessage()]
    assert [(r.levelno, r.getMessage()) for r in summary] == [
        (logging.INFO,
         "2 of 2 meshes labelled, 0 failed (1 segmented here, 1 already labelled)"),
    ]


def test_a_partial_run_closes_with_a_warning_summary(tmp_path, monkeypatch, model_file, caplog):
    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")

    def only_the_first(csv_path, output_dir, model_path, input_root, array_name, suffix,
                       device, fdi, num_workers=2):
        with open(csv_path, encoding="utf-8") as handle:
            meshes = [line.strip() for line in handle.read().splitlines()[1:] if line.strip()]
        write_surface(
            pipeline._predicted_path(output_dir, "crownseg_input", suffix, meshes[0],
                                     input_root),
            labelled=True,
        )

    monkeypatch.setattr(pipeline, "_run_engine", only_the_first)
    write_surface(tmp_path / "cohort" / "a.vtk")
    write_surface(tmp_path / "cohort" / "b.vtk")

    with caplog.at_level(logging.INFO, logger="sadt_crownseg"):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(model_file),
            output_dir=str(tmp_path / "out"),
        )

    summary = [r for r in caplog.records if "meshes labelled" in r.getMessage()]
    assert [(r.levelno, r.getMessage()) for r in summary] == [
        (logging.WARNING,
         "1 of 2 meshes labelled, 1 failed (1 segmented here, 0 already labelled)"),
    ]
    assert "mesh 2 of 2: segmentation failed" in caplog.text

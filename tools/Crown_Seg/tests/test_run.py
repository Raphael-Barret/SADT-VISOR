"""End-to-end tests for CrownSeg.

The segmentation engine is stubbed, so everything around it -- input discovery,
the already-segmented bypass, the output tree, the run report -- runs for real.
The network itself needs the `segmentation` extra (monai and embreex), which a
CI venv deliberately does not install. That is the whole reason it is imported
lazily, and `test_real_model_*` is what covers it when the extra is installed.
"""

import json
import os
from pathlib import Path

import pytest

from sadt_crownseg import run
from sadt_crownseg import pipeline
from sadt_crownseg.errors import ToolInputError, ToolUnavailableError


def write_surface(path, labelled=False, array_name="Universal_ID"):
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        points.InsertNextPoint(*coordinates)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(3)
    for point_id in range(3):
        polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName(array_name)
        for _ in range(3):
            labels.InsertNextValue(8)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


@pytest.fixture
def stub_engine_run(monkeypatch, tmp_path):
    """Write a labelled mesh wherever shapeaxi would have written one.

    The import probe is stubbed too. Replacing only `_run_engine` left a
    half-stubbed world: on a machine without the `segmentation` extra the probe
    still failed, so an already-segmented mesh came back `engine_unavailable`
    rather than `already_segmented` and the pass-through test failed for a
    reason that had nothing to do with pass-through. It passed locally, where
    the extra is installed, and failed in CI, where `uv sync --frozen` does not
    install extras. A fixture that stands in for a working engine has to stand
    in for all of it.
    """
    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    calls = []

    def fake_run(csv_path, output_dir, model_path, input_root, array_name, suffix,
                 device, fdi, num_workers=2):
        calls.append(
            {
                "csv": csv_path,
                "out": output_dir,
                "model": model_path,
                "root": input_root,
                "device": device,
                "fdi": fdi,
                "array_name": array_name,
            }
        )
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        with open(csv_path, encoding="utf-8") as handle:
            meshes = [line.strip() for line in handle.read().splitlines()[1:] if line.strip()]
        for mesh in meshes:
            destination = pipeline._predicted_path(
                output_dir, csv_stem, suffix, mesh, input_root
            )
            write_surface(destination, labelled=True, array_name=array_name)

    monkeypatch.setattr(pipeline, "_run_engine", fake_run)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")
    # The model is a required argument now, so the tests need a file that
    # exists -- segment_crowns refuses a path that does not.
    (tmp_path / "model.pth").write_bytes(b"not a real checkpoint")
    return calls


# ---------------------------------------------------------------------------
# Already-segmented detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("array_name", pipeline.LABEL_ARRAY_NAMES)
def test_every_known_label_array_counts_as_segmented(tmp_path, array_name):
    mesh = write_surface(tmp_path / "arch.vtk", labelled=True, array_name=array_name)
    assert pipeline.is_segmented(mesh) is True


def test_a_raw_mesh_is_not_segmented(tmp_path):
    assert pipeline.is_segmented(write_surface(tmp_path / "arch.vtk")) is False


def test_an_stl_can_never_be_segmented(tmp_path):
    """STL carries no point data by construction. The Slicer module copied a
    "segmented" .stl into its bypass folder and its CLI then globbed for .vtk
    only, so those meshes were silently never processed at all."""
    vtk = pytest.importorskip("vtk")

    source = write_surface(tmp_path / "arch.vtk")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(source)
    reader.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(tmp_path / "arch.stl"))
    writer.SetInputData(reader.GetOutput())
    writer.Write()

    assert pipeline.is_segmented(str(tmp_path / "arch.stl")) is False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discovery_is_recursive(tmp_path):
    write_surface(tmp_path / "cohort" / "a" / "one.vtk")
    write_surface(tmp_path / "cohort" / "b" / "two.vtk")

    meshes = pipeline.discover_meshes(str(tmp_path / "cohort"))
    assert len(meshes) == 2


def test_a_file_of_the_wrong_kind_is_a_422(tmp_path):
    (tmp_path / "notes.txt").write_text("not a mesh")
    with pytest.raises(ToolInputError):
        pipeline.discover_meshes(str(tmp_path / "notes.txt"))


def test_an_empty_folder_is_a_422(tmp_path):
    (tmp_path / "cohort").mkdir()
    with pytest.raises(ToolInputError):
        pipeline.discover_meshes(str(tmp_path / "cohort"))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_a_raw_mesh_is_segmented_and_reported(tmp_path, stub_engine_run):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out")
    )

    assert report["summary"] == {
        "total": 1, "segmented": 1, "already_segmented": 0, "failed": 0,
        "engine_unavailable": 0,
    }
    assert len(report["segmented_meshes"]) == 1
    assert pipeline.is_segmented(report["segmented_meshes"][0])


def test_an_already_segmented_mesh_is_passed_through(tmp_path, stub_engine_run):
    """Minutes of GPU time per mesh, for an array that is already there."""
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out")
    )

    assert stub_engine_run == []  # the network was never invoked
    assert report["summary"]["already_segmented"] == 1
    assert pipeline.is_segmented(report["segmented_meshes"][0])


def test_skip_segmented_false_re_runs_the_network(tmp_path, stub_engine_run):
    write_surface(tmp_path / "cohort" / "arch.vtk", labelled=True)

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out"),
        skip_segmented=False,
    )
    assert len(stub_engine_run) == 1
    assert report["summary"]["segmented"] == 1


def test_a_batch_keeps_its_tree(tmp_path, stub_engine_run):
    write_surface(tmp_path / "cohort" / "siteA" / "arch.vtk")
    write_surface(tmp_path / "cohort" / "siteB" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out")
    )

    # Two patients with the same base name. Flattened, one would overwrite the
    # other and the caller would silently get half its cohort back.
    assert len(report["segmented_meshes"]) == 2
    assert len(set(report["segmented_meshes"])) == 2
    assert sorted(report["cases"]) == [
        os.path.join("siteA", "arch.vtk"), os.path.join("siteB", "arch.vtk")
    ]


def test_the_input_csv_is_written_under_the_output_dir_and_removed(tmp_path, stub_engine_run):
    """The Slicer module wrote it into the extension's own source folder,
    which breaks a read-only install and leaves behind a file pointing at the
    patient's data."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out")
    )
    csv_path = stub_engine_run[0]["csv"]
    assert csv_path.startswith(str(tmp_path / "out"))
    assert not os.path.exists(csv_path), "the work dir does not survive the run"


def test_the_run_report_lands_beside_the_results(tmp_path, stub_engine_run):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    returned = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out")
    )
    written = json.loads(
        open(os.path.join(str(tmp_path / "out"), "run_report.json"), encoding="utf-8").read()
    )
    assert written["numbering"] == "Universal"
    assert written["array_name"] == pipeline.DEFAULT_ARRAY_NAME
    assert written == returned


def test_fdi_numbering_is_passed_through(tmp_path, stub_engine_run):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out"), fdi=True
    )
    assert stub_engine_run[0]["fdi"] is True
    assert report["numbering"] == "FDI"


def test_a_missing_checkpoint_is_reported_as_an_input_error(tmp_path, stub_engine_run):
    """The model is an argument now, not a name looked up in a data store."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(ToolInputError, match="checkpoint not found"):
        pipeline.segment_crowns(
            input_path=str(tmp_path / "cohort"),
            model_path=str(tmp_path / "absent.pth"),
            output_dir=str(tmp_path / "out"),
        )


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_returns_the_output_directory_it_was_given(tmp_path, stub_engine_run):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    output = run(
        meshes=tmp_path / "cohort",
        model=tmp_path / "model.pth",
        output_dir=tmp_path / "out",
    )

    assert output == tmp_path / "out"
    assert (output / "run_report.json").is_file()


def test_run_writes_nothing_outside_the_output_directory(tmp_path, stub_engine_run):
    write_surface(tmp_path / "cohort" / "arch.vtk")
    before = sorted(p for p in tmp_path.rglob("*") if p.is_file())

    run(meshes=tmp_path / "cohort", model=tmp_path / "model.pth", output_dir=tmp_path / "out")

    after = sorted(
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(tmp_path / "out")
    )
    assert after == before


def test_the_work_dir_does_not_survive_the_run(tmp_path, stub_engine_run):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    output = run(meshes=tmp_path / "cohort", model=tmp_path / "model.pth",
                 output_dir=tmp_path / "out")

    assert not (output / pipeline.WORK_DIRNAME).exists()


def test_the_report_lists_what_a_caller_sequencing_ali_needs(tmp_path, stub_engine_run):
    """The server runs CrownSeg then ALI, so the report has to name the meshes
    that now carry labels -- ALI no longer imports this tool to find out."""
    write_surface(tmp_path / "cohort" / "raw.vtk")
    write_surface(tmp_path / "cohort" / "done.vtk", labelled=True)

    output = run(meshes=tmp_path / "cohort", model=tmp_path / "model.pth",
                 output_dir=tmp_path / "out")

    with open(output / "run_report.json") as handle:
        report = json.load(handle)

    assert len(report["segmented_meshes"]) == 2
    assert all(pipeline.is_segmented(path) for path in report["segmented_meshes"])
    assert all(Path(path).is_relative_to(output) for path in report["segmented_meshes"])


# ---------------------------------------------------------------------------
# the real engine
# ---------------------------------------------------------------------------

REAL_MODEL = os.environ.get("SADT_CROWNSEG_MODEL")
REAL_MESH = os.environ.get("SADT_CROWNSEG_MESH")


def test_a_venv_without_the_extra_says_how_to_install_it(tmp_path, monkeypatch):
    """`uv sync` alone cannot run the network, and has to say so.

    This is the state CI is in and the state a fresh clone is in, so the
    message is the only thing standing between a maintainer and a bare
    ModuleNotFoundError from inside the renderer.
    """
    if _engine_installed():
        pytest.skip("the segmentation extra is installed in this venv")

    with pytest.raises(ToolUnavailableError, match="uv sync --extra segmentation"):
        pipeline._import_engine()


def _engine_installed() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in ("monai", "embreex"))


@pytest.mark.gpu
@pytest.mark.models
@pytest.mark.skipif(
    not (REAL_MODEL and REAL_MESH),
    reason="set SADT_CROWNSEG_MODEL and SADT_CROWNSEG_MESH (see tests/data/README.md)",
)
def test_real_model_labels_a_real_mesh(tmp_path):
    """The real checkpoint through the real engine, on a real arch."""
    output = run(
        meshes=Path(REAL_MESH),
        model=Path(REAL_MODEL),
        output_dir=tmp_path / "out",
        skip_segmented=False,
    )

    with open(output / "run_report.json") as handle:
        report = json.load(handle)

    assert report["summary"]["segmented"] == 1
    assert report["summary"]["failed"] == 0
    produced = report["segmented_meshes"][0]
    assert pipeline.is_segmented(produced)


# ---------------------------------------------------------------------------
# The checkpoint, when nobody names one
# ---------------------------------------------------------------------------

def test_the_installed_checkpoint_is_used_when_none_is_named(tmp_path, stub_engine_run):
    """A panel with no model field sends none, and the tool finds its own.

    Left required, this was not merely a field too many: AREG IOS and IOSCBCT
    call `label_crowns(sup, root, crown_model or "")`, `crown_model` is optional
    on their own schemas, and the supervised call therefore omits `model` --
    which used to be `TypeError: run() missing 1 required positional argument`.
    """
    write_surface(tmp_path / "cohort" / "arch.vtk")
    models = tmp_path / "data" / "Crown_Seg" / "models"
    models.mkdir(parents=True)
    (models / pipeline.PUBLISHED_CHECKPOINT).write_bytes(b"not a real checkpoint")

    output = run(meshes=tmp_path / "cohort", output_dir=tmp_path / "out",
                 data_root=tmp_path / "data")

    report = json.loads((output / "run_report.json").read_text())
    assert report["checkpoint"] == pipeline.PUBLISHED_CHECKPOINT
    assert stub_engine_run[0]["model"].endswith(pipeline.PUBLISHED_CHECKPOINT)


def test_a_named_checkpoint_still_wins_over_the_installed_one(tmp_path, stub_engine_run):
    """Naming one is pinning which weights ran, and is obeyed."""
    write_surface(tmp_path / "cohort" / "arch.vtk")
    models = tmp_path / "data" / "Crown_Seg" / "models"
    models.mkdir(parents=True)
    (models / pipeline.PUBLISHED_CHECKPOINT).write_bytes(b"not a real checkpoint")

    run(meshes=tmp_path / "cohort", model=tmp_path / "model.pth",
        output_dir=tmp_path / "out", data_root=tmp_path / "data")

    assert stub_engine_run[0]["model"] == str(tmp_path / "model.pth")


def test_no_model_and_no_data_root_says_which_of_the_two_is_missing(tmp_path, stub_engine_run):
    """A checkout running this by hand, rather than a server publishing a data
    root."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(ToolInputError, match="no data root"):
        run(meshes=tmp_path / "cohort", output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# The array name follows the numbering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("numbering,expected", [
    ("Universal", pipeline.DEFAULT_ARRAY_NAME),
    ("FDI", pipeline.FDI_ARRAY_NAME),
])
def test_the_array_is_named_after_the_numbering(tmp_path, stub_engine_run, numbering, expected):
    """shapeaxi's `ConvertFDI` rewrites the values in place under the SAME array
    name, so asking for FDI produced an array called `Universal_ID` holding FDI
    numbers. The two systems overlap almost completely -- FDI 18 is the upper
    right third molar where Universal 18 is the lower left second molar -- so a
    consumer did not fail, it read a different tooth."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    output = run(meshes=tmp_path / "cohort", model=tmp_path / "model.pth",
                 output_dir=tmp_path / "out", numbering=numbering)

    assert stub_engine_run[0]["array_name"] == expected
    assert json.loads((output / "run_report.json").read_text())["array_name"] == expected


def test_an_fdi_mesh_carries_no_name_the_consumers_know(tmp_path):
    """The point of the rename: ALI's IOS landmarks, ASO, AREG and FlexReg all
    look for the Universal spellings, so an FDI mesh is REFUSED there rather
    than read as another set of teeth. None of them speaks FDI, and every
    legacy caller passed `fdi: 0`."""
    consumed = ("Universal_ID", "PredictedID", "UniversalID")

    assert pipeline.FDI_ARRAY_NAME not in consumed
    # Crown_Seg itself must still recognise its own output, or a second run
    # would re-predict a batch it has already labelled.
    assert pipeline.FDI_ARRAY_NAME in pipeline.LABEL_ARRAY_NAMES


def test_a_named_array_still_wins(tmp_path, stub_engine_run):
    """A caller that names one is choosing, and is obeyed -- which is what
    keeps a request written against the old default working unchanged."""
    write_surface(tmp_path / "cohort" / "arch.vtk")

    run(meshes=tmp_path / "cohort", model=tmp_path / "model.pth",
        output_dir=tmp_path / "out", numbering="FDI", array_name="Universal_ID")

    assert stub_engine_run[0]["array_name"] == "Universal_ID"


# ---------------------------------------------------------------------------
# Progress -- counted off the outputs shapeaxi writes, one per finished mesh
# ---------------------------------------------------------------------------

def _events(events_file):
    """Progress records and log records, apart. Logs carry `kind: log`."""
    records = [json.loads(line) for line in
               Path(events_file).read_text().splitlines() if line]
    return ([r for r in records if r.get("kind") != "log"],
            [r for r in records if r.get("kind") == "log"])


def _engine_writing_one_mesh_at_a_time(monkeypatch, events_file, write_only=None):
    """An engine that writes its outputs in order, the way shapeaxi does, and
    after each one waits until the watcher has seen it -- so the test does not
    depend on how fast this machine is."""
    import time

    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")
    monkeypatch.setattr(pipeline, "WATCH_INTERVAL", 0.01)

    def fake_run(csv_path, output_dir, model_path, input_root, array_name, suffix,
                 device, fdi, num_workers=2):
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        with open(csv_path, encoding="utf-8") as handle:
            meshes = [line.strip() for line in handle.read().splitlines()[1:] if line.strip()]
        for done, mesh in enumerate(meshes[:write_only], start=1):
            write_surface(
                pipeline._predicted_path(output_dir, csv_stem, suffix, mesh, input_root),
                labelled=True, array_name=array_name,
            )
            if done == len(meshes):
                break
            wanted = "mesh {} of {}".format(done + 1, len(meshes))
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if wanted in Path(events_file).read_text():
                    break
                time.sleep(0.01)

    monkeypatch.setattr(pipeline, "_run_engine", fake_run)


def test_progress_counts_the_meshes_shapeaxi_has_finished(tmp_path, monkeypatch):
    """shapeaxi takes the batch in one call and says nothing on the way, but
    it writes each mesh as soon as that mesh is done. Counting those files is
    a count, not an estimate, so the bar moves per mesh and never ahead."""
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _engine_writing_one_mesh_at_a_time(monkeypatch, events_file)
    for name in ("Smith_John", "Jones_Mary", "Brown_Ann"):
        write_surface(tmp_path / "cohort" / (name + ".vtk"))
    (tmp_path / "model.pth").write_bytes(b"not a real checkpoint")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out"),
    )

    assert report["summary"]["segmented"] == 3
    events, logs = _events(events_file)
    assert [event["message"] for event in events] == [
        "mesh 1 of 3", "mesh 2 of 3", "mesh 3 of 3",
    ]
    fractions = [event["fraction"] for event in events]
    assert fractions == sorted(fractions)
    assert all(0.0 <= fraction < 1.0 for fraction in fractions)
    assert logs == []
    text = events_file.read_text()
    assert not any(name in text for name in ("Smith", "Jones", "Brown", ".vtk"))


def test_a_mesh_shapeaxi_did_not_write_is_logged_to_the_clinician_by_position(
    tmp_path, monkeypatch
):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    _engine_writing_one_mesh_at_a_time(monkeypatch, events_file, write_only=1)
    write_surface(tmp_path / "cohort" / "a_Smith_John.vtk")
    write_surface(tmp_path / "cohort" / "b_Jones_Mary.vtk")
    (tmp_path / "model.pth").write_bytes(b"not a real checkpoint")

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=str(tmp_path / "model.pth"),
        output_dir=str(tmp_path / "out"),
    )

    assert report["summary"]["failed"] == 1
    _, logs = _events(events_file)
    assert [(log["level"], log["audience"], log["message"]) for log in logs] == [
        ("warning", "user", "mesh 2 of 2 produced no segmentation"),
    ]
    assert "Jones" not in events_file.read_text()


def test_the_watcher_never_counts_an_output_left_by_an_earlier_run(tmp_path, monkeypatch):
    """A file already on disk says nothing about THIS run; counting it would
    put a mesh on the bar that is still to come."""
    import time

    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    stale = tmp_path / "stale.vtk"
    stale.write_bytes(b"")
    fresh = tmp_path / "fresh.vtk"

    with pipeline._counting_outputs([str(stale), str(fresh), str(tmp_path / "x.vtk")],
                                    "mesh", interval=0.01):
        time.sleep(0.05)
        assert not events_file.exists()
        fresh.write_bytes(b"")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not events_file.exists():
            time.sleep(0.01)

    events, _ = _events(events_file)
    assert [event["message"] for event in events] == ["mesh 2 of 3"]

"""Fixtures shared by the CrownSeg suites.

The network cannot run in CI -- it lives behind the `segmentation` extra,
because it needs monai and embreex, which a CI venv does not install -- so the engine is stubbed and everything
around it runs for real: VTK writes the meshes, the discovery walks a real
tree, the report is the real report. `_import_engine` is stubbed too:
replacing only `_run_engine` leaves a half-stubbed world where the import
probe still fails and an already-labelled mesh comes back `engine_unavailable`.
"""

import os

import pytest

from sadt_crownseg import pipeline


def write_surface(path, labelled=False, array_name="Universal_ID", points=3):
    """A minimal triangle, optionally carrying a per-point label array."""
    vtk = pytest.importorskip("vtk")

    coordinates = [(0, 0, 0), (1, 0, 0), (0, 1, 0)][:points]
    vtk_points = vtk.vtkPoints()
    for coordinate in coordinates:
        vtk_points.InsertNextPoint(*coordinate)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(len(coordinates))
    for point_id in range(len(coordinates)):
        polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(vtk_points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName(array_name)
        for _ in coordinates:
            labels.InsertNextValue(8)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


def write_stl(path, source_vtk=None, tmp_path=None):
    """The same triangle as an .stl -- which by construction has no point data."""
    vtk = pytest.importorskip("vtk")

    source = source_vtk or write_surface(tmp_path / "_source.vtk")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(source))
    reader.Update()

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(reader.GetOutput())
    writer.Write()
    return str(path)


@pytest.fixture
def model_file(tmp_path):
    """A checkpoint that exists. `segment_crowns` refuses one that does not."""
    path = tmp_path / "model.pth"
    path.write_bytes(b"not a real checkpoint")
    return path


@pytest.fixture
def stub_engine(monkeypatch, model_file):
    """Write a labelled mesh wherever shapeaxi would have written one.

    Returns the list of calls, so a test can assert on the namespace the engine
    was handed as well as on what came out.
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
                "array_name": array_name,
                "suffix": suffix,
                "device": device,
                "fdi": fdi,
                "num_workers": num_workers,
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
    return calls


@pytest.fixture
def failing_engine(monkeypatch, model_file):
    """An engine that runs but produces no output for any mesh."""
    monkeypatch.setattr(pipeline, "_import_engine", lambda: None)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")
    monkeypatch.setattr(pipeline, "_run_engine", lambda **kwargs: None)


@pytest.fixture
def absent_engine(monkeypatch, model_file):
    """The state a venv without the `segmentation` extra is in."""
    def refuse():
        raise pipeline.ToolUnavailableError(
            "CrownSeg's engine is an optional extra. Install it with "
            "`uv sync --extra segmentation` in tools/Crown_Seg. (missing: monai)"
        )

    monkeypatch.setattr(pipeline, "_import_engine", refuse)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested=None: "cpu")

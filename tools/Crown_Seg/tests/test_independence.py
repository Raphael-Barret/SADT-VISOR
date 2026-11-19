"""This tool runs without shapeaxi, pytorch3d or anything they pull in.

That is the reason the engine exists, so it is tested rather than intended: the
import chain is walked in a fresh interpreter in which those packages are made
unimportable (a meta-path finder that raises on them), and a whole run is made
there too, on a synthetic mesh and a randomly initialised network.

Needs monai and embreex; a venv that lacks them skips the file.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("embreex")
pytest.importorskip("vtk")

TOOL_DIR = Path(__file__).resolve().parents[1]
SRC = TOOL_DIR / "src"

# What `pip install shapeaxi` brings, and what open3d would add. None of them
# may be needed, and none may be imported.
FORBIDDEN = (
    "shapeaxi", "pytorch3d", "torchvision", "lightning", "pytorch_lightning", "ocnn",
    "open3d", "pandas", "itk", "SimpleITK", "nibabel", "fslpy", "h5py", "sklearn",
)

BLOCKER = textwrap.dedent(
    """
    import importlib.abc, sys

    FORBIDDEN = {forbidden!r}

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in FORBIDDEN:
                raise ImportError("blocked for this test: " + name, name=name)

    sys.meta_path.insert(0, Blocker())
    sys.path.insert(0, {src!r})
    """
).format(forbidden=FORBIDDEN, src=str(SRC))


def run_blocked(body, tmp_path):
    return subprocess.run(
        [sys.executable, "-I", "-c", BLOCKER + textwrap.dedent(body)],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )


def test_the_engine_imports_with_shapeaxi_and_pytorch3d_unimportable(tmp_path):
    finished = run_blocked(
        """
        from sadt_crownseg import engine, pipeline

        assert pipeline._import_engine() is engine
        leaked = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)
        assert not leaked, leaked
        print("ok")
        """,
        tmp_path,
    )

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip().endswith("ok")


def test_a_whole_run_works_with_them_unimportable(tmp_path):
    """The public entry point, on a mesh and a checkpoint made here, from the
    scan to the labelled file and the report."""
    finished = run_blocked(
        f"""
        import json
        from pathlib import Path

        import torch, vtk
        from sadt_crownseg import engine, run

        engine.IMAGE_SIZE = 48          # the network is fully convolutional; this is faster

        work = Path({str(tmp_path)!r})
        source = vtk.vtkSphereSource()
        source.SetRadius(20)
        source.SetThetaResolution(20)
        source.SetPhiResolution(20)
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(source.GetOutputPort())
        (work / "cohort").mkdir()
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(work / "cohort" / "scan.vtk"))
        writer.SetInputConnection(triangles.GetOutputPort())
        writer.Write()

        torch.manual_seed(0)
        torch.save(engine.create_network().state_dict(), work / "toy_val-loss0.1.pth")

        out = run(meshes=work / "cohort", model=work / "toy_val-loss0.1.pth",
                  output_dir=work / "out", device="cpu")
        report = json.loads((out / "run_report.json").read_text())

        assert report["summary"] == {{"total": 1, "segmented": 1, "already_segmented": 0,
                                      "engine_unavailable": 0, "failed": 0}}, report
        produced = Path(report["segmented_meshes"][0])
        assert produced == work / "out" / "crownseg_input_Seg" / "scan_Seg.vtk", produced
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(produced))
        reader.Update()
        assert reader.GetOutput().GetPointData().GetArray("Universal_ID") is not None
        leaked = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)
        assert not leaked, leaked
        print("ok")
        """,
        tmp_path,
    )

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip().endswith("ok")


def test_no_source_file_names_a_forbidden_package():
    """The static twin of the two runs above: a lazy import inside a function
    would pass them and fail on the first call that reaches it."""
    offenders = {}
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in FORBIDDEN:
                    offenders.setdefault(path.name, set()).add(name)

    assert offenders == {}


def test_open3d_is_not_a_dependency_of_the_tool():
    import tomllib

    pyproject = tomllib.loads((TOOL_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(pyproject["project"]["dependencies"])
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        declared.extend(extra)

    for entry in declared:
        name = entry.split("==")[0].split(">")[0].split("<")[0].split("[")[0].strip().lower()
        assert name.replace("-", "_") not in {n.lower() for n in FORBIDDEN}, entry

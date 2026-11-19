"""The seam between this tool and its engine.

The pipeline hands the engine a csv of meshes, one network load, and the paths
the results go to; the engine does the rest. What is pinned here is exactly what
crosses that line: which mesh goes where, with which array name, numbering and
device, and what the caller sees of the engine's output and failures.

The engine itself is replaced by a stand-in, so none of this needs monai,
embreex or a checkpoint.
"""

import os
import sys
import types

import pytest

from sadt_crownseg import pipeline
from sadt_crownseg.errors import ToolInputError, ToolUnavailableError


class FakeEngine:
    """Records what the pipeline asked of the engine, and writes nothing."""

    def __init__(self):
        self.loaded = []
        self.segmented = []
        self.network = types.SimpleNamespace(placed=None, evaluating=False)
        self.network.to = self._to
        self.network.eval = self._eval

    def _to(self, device):
        self.network.placed = device
        return self.network

    def _eval(self):
        self.network.evaluating = True
        return self.network

    def load_network(self, path):
        self.loaded.append(path)
        return self.network

    def camera_rig(self):
        return "rig"

    def segment_mesh(self, network, rig, mesh_path, output_path, array_name, fdi=False, device="cpu"):
        assert network is self.network and rig == "rig"
        self.segmented.append(
            {"mesh": mesh_path, "output": output_path, "array_name": array_name,
             "fdi": fdi, "device": device}
        )


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(pipeline, "_import_engine", lambda: fake)
    return fake


def invoke(tmp_path, meshes=("a.vtk",), **overrides):
    csv_path = tmp_path / "crownseg_input.csv"
    csv_path.write_text("surf\n" + "".join(f"{tmp_path / 'cohort' / name}\n" for name in meshes))
    arguments = dict(
        csv_path=str(csv_path),
        output_dir=str(tmp_path / "out"),
        model_path=str(tmp_path / "model.pth"),
        input_root=str(tmp_path / "cohort"),
        array_name="Universal_ID",
        suffix="Seg",
        device="cpu",
        fdi=False,
        num_workers=2,
    )
    arguments.update(overrides)
    pipeline._run_engine(**arguments)


# ---------------------------------------------------------------------------
# What the engine is asked to do
# ---------------------------------------------------------------------------

def test_every_mesh_of_the_csv_goes_to_the_path_the_report_will_look_at(tmp_path, engine):
    """The report finds each result by recomputing `_predicted_path`, so the
    engine has to be given exactly that path, and the folder layout
    `<out>/<csv stem>_<suffix>/<tree>` with it."""
    invoke(tmp_path, meshes=("a.vtk", "sub/b.stl"))

    expected = [
        pipeline._predicted_path(
            str(tmp_path / "out"), "crownseg_input", "Seg", str(tmp_path / "cohort" / name),
            str(tmp_path / "cohort"),
        )
        for name in ("a.vtk", "sub/b.stl")
    ]
    assert [call["output"] for call in engine.segmented] == expected
    assert expected[0].endswith(os.path.join("crownseg_input_Seg", "a_Seg.vtk"))
    assert expected[1].endswith(os.path.join("crownseg_input_Seg", "sub", "b_Seg.vtk"))


def test_the_checkpoint_is_loaded_once_for_the_whole_batch(tmp_path, engine):
    invoke(tmp_path, meshes=("a.vtk", "b.vtk", "c.vtk"))

    assert engine.loaded == [str(tmp_path / "model.pth")]
    assert len(engine.segmented) == 3


def test_the_network_is_placed_on_the_resolved_device_and_put_in_eval_mode(tmp_path, engine):
    invoke(tmp_path, device="cuda:0")

    assert engine.network.placed == "cuda:0"
    assert engine.network.evaluating is True
    assert engine.segmented[0]["device"] == "cuda:0"


def test_the_array_name_and_the_numbering_are_passed_through_verbatim(tmp_path, engine):
    invoke(tmp_path, array_name="FDI_ID", fdi=True)

    assert engine.segmented[0]["array_name"] == "FDI_ID"
    assert engine.segmented[0]["fdi"] is True


def test_the_workers_argument_is_accepted_and_changes_nothing(tmp_path, engine):
    """It sized shapeaxi's DataLoader. The engine reads in the calling process,
    but the argument is published, so a request that sends it must still work."""
    invoke(tmp_path, num_workers=7)
    invoke(tmp_path, num_workers=0)

    assert len(engine.segmented) == 2


# ---------------------------------------------------------------------------
# What the engine prints, and what it raises
# ---------------------------------------------------------------------------

def test_the_engines_stdout_never_reaches_the_process_output(tmp_path, monkeypatch, capsys):
    """A library under the engine may print the path of the scan it is reading,
    and that path carries the patient's own file name, which must not reach a
    log."""
    chatty = FakeEngine()
    original = chatty.segment_mesh

    def segment_mesh(*args, **kwargs):
        print("Saving results to /data/patients/Smith_John_T1_Seg.vtk")
        return original(*args, **kwargs)

    chatty.segment_mesh = segment_mesh
    monkeypatch.setattr(pipeline, "_import_engine", lambda: chatty)

    invoke(tmp_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Smith_John" not in captured.err


def test_a_real_failure_from_the_engine_is_left_untouched(tmp_path, monkeypatch):
    """Swallowing stdout must not swallow the exception with it: the run has to
    fail, and the server has to see why."""
    failing = FakeEngine()

    def segment_mesh(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    failing.segment_mesh = segment_mesh
    monkeypatch.setattr(pipeline, "_import_engine", lambda: failing)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        invoke(tmp_path)


def test_a_refused_checkpoint_is_not_rewrapped(tmp_path, monkeypatch):
    """`ToolInputError` is what the server answers 422 with; wrapping it would
    turn a wrong file into a 500."""
    refusing = FakeEngine()

    def load_network(path):
        raise ToolInputError("not the published checkpoint")

    refusing.load_network = load_network
    monkeypatch.setattr(pipeline, "_import_engine", lambda: refusing)

    with pytest.raises(ToolInputError, match="not the published"):
        invoke(tmp_path)


def test_stdout_is_restored_after_the_call(tmp_path, engine, capsys):
    """A redirect left in place would silence the whole worker process."""
    invoke(tmp_path)
    print("still here")

    assert capsys.readouterr().out == "still here\n"


def test_stdout_is_restored_even_when_the_engine_raises(tmp_path, monkeypatch, capsys):
    failing = FakeEngine()

    def segment_mesh(*args, **kwargs):
        print("half a mesh")
        raise RuntimeError("boom")

    failing.segment_mesh = segment_mesh
    monkeypatch.setattr(pipeline, "_import_engine", lambda: failing)

    with pytest.raises(RuntimeError):
        invoke(tmp_path)
    print("still here")

    assert capsys.readouterr().out == "still here\n"


# ---------------------------------------------------------------------------
# A venv without the segmentation extra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("package", ["monai", "embreex"])
def test_an_absent_engine_dependency_says_how_to_install_it(monkeypatch, package):
    """`uv sync` alone cannot run the network, and has to say so. The message
    is the only thing between a maintainer and a bare ModuleNotFoundError from
    inside the renderer -- and a 501 body a clinician reads."""
    pytest.importorskip("torch")
    pytest.importorskip("vtk")
    if package == "embreex":
        pytest.importorskip("monai")   # else monai is what the probe reports first
    # `from . import engine` finds the attribute the package already holds
    # before it looks at sys.modules, so both have to go.
    import sadt_crownseg

    for name in ("engine", "render_native"):
        monkeypatch.delitem(sys.modules, f"sadt_crownseg.{name}", raising=False)
        monkeypatch.delattr(sadt_crownseg, name, raising=False)
    monkeypatch.setitem(sys.modules, package, None)
    if package == "embreex":
        monkeypatch.setitem(sys.modules, "embreex.rtcore_scene", None)
        monkeypatch.setitem(sys.modules, "embreex.mesh_construction", None)

    with pytest.raises(ToolUnavailableError) as failure:
        pipeline._import_engine()

    message = str(failure.value)
    assert "uv sync --extra segmentation" in message
    assert "tools/Crown_Seg" in message
    assert f"missing: {package}" in message


def test_the_import_probe_returns_the_engine_module_when_everything_is_installed():
    pytest.importorskip("monai")
    pytest.importorskip("embreex")

    from sadt_crownseg import engine

    assert pipeline._import_engine() is engine


# ---------------------------------------------------------------------------
# The error classes the server maps to status codes
# ---------------------------------------------------------------------------

def test_the_input_error_is_the_one_the_server_answers_422_with():
    """`dispatch.py` maps by exception class NAME, there being no shared
    package to define a base class in. The base class matters too: a plain
    ValueError escaping the tool is answered the same way."""
    assert ToolInputError.__name__ == "ToolInputError"
    assert issubclass(ToolInputError, ValueError)


def test_the_unavailable_error_is_the_one_the_server_answers_501_with():
    assert ToolUnavailableError.__name__ == "ToolUnavailableError"
    assert issubclass(ToolUnavailableError, RuntimeError)
    assert not issubclass(ToolUnavailableError, ValueError)

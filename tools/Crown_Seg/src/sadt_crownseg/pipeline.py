"""Crown segmentation on intraoral surface scans.

Labels every tooth of a mesh with its Universal number, as a point-data array
on a copy of the surface. That array is the precondition for ALI's IOS landmark
identification and for the IOS modes of ASO, AREG and FlexReg, which is why
this is its own tool rather than a helper inside any of them -- and, since the
split, why the server sequences CrownSeg before ALI instead of ALI importing
this module.

The method is shapeaxi's `dentalmodelseg` (the Slicer modules ran its
executable), but shapeaxi itself is no longer called: it imports pytorch3d,
which has no Windows wheel and no Metal kernel. `engine.py` carries the same
pipeline on torch, monai, VTK and Embree, and this module only prepares what it
needs (the mesh list, the output paths, the device) and reports what came out.

What the move out of the server changed: scratch space lives under the caller's
`output_dir`, `segment_crowns()` returns the run report rather than a
`CrownSegRun`, zip extraction is gone (the server unpacks archives before
`run()` is called), the model is a required argument rather than a name looked
up in the data store, and the GPU semaphore is gone because each call is its
own process.
"""

import contextlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
import time

from . import progress
from .errors import ToolInputError, ToolUnavailableError

logger = logging.getLogger(__name__)


# The tool's own name, derived rather than written down. `describe.py` publishes
# `tool_dir.name` as the schema's `name`, so taking it from the same place is
# what keeps a report and the schema from disagreeing -- which they already did:
# this file said "CrownSeg" and "BatchDentalSeg" long after the folders were
# renamed to the underscored convention, and nothing noticed because nothing
# reads the field yet.
#
# `parents[2]` is <tool>/src/<package>/thisfile.py -> <tool>.
TOOL_NAME = Path(__file__).resolve().parents[2].name

# Extensions the engine's reader handles and this tool therefore discovers.
SURFACE_EXTENSIONS = (".vtk", ".stl")

# Point-data array names that already carry per-tooth labels. A mesh holding
# any of them is segmented and is passed through untouched -- re-running the
# network on it would cost minutes and change nothing.
LABEL_ARRAY_NAMES = ("PredictedID", "UniversalID", "Universal_ID", "FDI_ID")

DEFAULT_ARRAY_NAME = "Universal_ID"
FDI_ARRAY_NAME = "FDI_ID"
DEFAULT_SUFFIX = "Seg"

# The array a run writes to, named after the numbering that went into it.
#
# **The name has to follow the values, and this is not tidiness.** shapeaxi's
# `ConvertFDI` converts the integers in place and writes them back under the
# SAME array name, so asking for FDI used to produce an array literally called
# `Universal_ID` holding FDI numbers. The two systems overlap across almost
# their whole range -- Universal 1..32 against FDI 11..18, 21..28, 31..38,
# 41..48 -- so a consumer reading that array does not fail, it reads a
# different tooth: FDI 18 is the upper right third molar where Universal 18 is
# the lower left second molar, and FDI 33 (lower left canine) is the value
# shapeaxi reserves for GUM in Universal.
#
# Naming the FDI array differently is what turns that into a refusal. Every
# tool that consumes these meshes -- ALI's IOS landmarks, ASO, AREG and
# FlexReg -- looks for `Universal_ID`, `PredictedID` or `UniversalID`, so an
# FDI mesh now carries no name they know and they say so, instead of
# registering a patient onto the wrong teeth. That is the right answer: none of
# them speaks FDI, and every legacy caller passed `fdi: 0`.
_ARRAY_NAMES = {"Universal": DEFAULT_ARRAY_NAME, "FDI": FDI_ARRAY_NAME}


def array_name_for(numbering: str) -> str:
    """The array name that goes with a numbering system."""
    return _ARRAY_NAMES.get(numbering, DEFAULT_ARRAY_NAME)

WORK_DIRNAME = ".crownseg_work"

# How often, in seconds, the output tree is looked at while the engine runs. One
# mesh takes about 52 s on the card, so two seconds is fine-grained enough to
# move the bar promptly and coarse enough to cost nothing.
WATCH_INTERVAL = 2.0

# The published crown-segmentation checkpoint, and the token every one of them
# is named with. Fly-by-CNN publishes its weights under the training run that
# produced them -- `<date>_val-loss<number>.pth` -- and that name is what
# `scripts/data-manifest.yml` stages, byte for byte, under every tool that
# needs it.
PUBLISHED_CHECKPOINT = "07-21-22_val-loss0.169.pth"
CROWN_CHECKPOINT_MARKER = "val-loss"

_INSTALL_HINT = (
    "CrownSeg's engine is an optional extra (monai and embreex). Install it with "
    "`uv sync --extra segmentation` in tools/Crown_Seg. A deployment image "
    "builds it with `--all-extras`; if this appears there, the image was built "
    "without them."
)


def _import_engine():
    """Import the segmentation engine, lazily.

    Deferred so that importing this package -- which `scripts/describe.py` does
    on every CI run to publish the schema -- costs nothing. It also means a
    venv without the extra still loads and reports the schema; only an actual
    segmentation fails, naming what is missing.

    Two things can be missing and they are probed together: monai, which the
    engine imports at load time, and embreex, which the renderer would only
    import on its first ray. Left to the first ray, a missing embreex would
    pass this probe, report `engine_available` and then fail the run.
    """
    try:
        from . import engine, render_native

        render_native.require_embree()
    except ImportError as exc:
        raise ToolUnavailableError(
            "{} (missing: {})".format(_INSTALL_HINT, exc.name or "monai")
        ) from exc
    except render_native.ToolUnavailableError as exc:
        raise ToolUnavailableError("{} ({})".format(_INSTALL_HINT, str(exc).rsplit("(", 1)[-1].rstrip(")"))) from exc

    return engine


def resolve_device(requested: str = None) -> str:
    """The device the network runs on: "cpu", "cuda:0" or, on a Mac, "mps".

    CUDA is always the first card: a tool is one process with whatever the
    server made visible to it. A device that cannot be had falls back to the CPU
    with a warning rather than failing the run. MPS is honoured on macOS only;
    shapeaxi accepted nothing but "cpu" and "cuda:0", so a request for "mps"
    used to run on the CPU without a word, which the engine no longer does.
    """
    import torch

    wanted = (requested or "cpu").strip().lower()
    if wanted.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda:0"
        logger.warning("device=%s requested but CUDA is unavailable; falling back to CPU", wanted)
        progress.log(
            "a GPU was requested but none is visible; segmenting on the CPU",
            "warning",
        )
    elif wanted == "mps":
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        logger.warning("device=mps requested but MPS is unavailable; falling back to CPU")
        progress.log(
            "the Apple GPU was requested but is not available; segmenting on the CPU",
            "warning",
        )
    return "cpu"


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def _is_crown_checkpoint(name: str) -> bool:
    """Whether this file name is a crown-segmentation checkpoint.

    **The name is the only thing that can answer this, and that is measured
    rather than assumed.** The obvious alternative -- open the checkpoint and
    look at what it holds -- does not separate it from its neighbours: this
    network and ALI_IOS's landmark networks are both monai UNets, and their
    state dicts start with the very same key
    (`model.0.conv.unit0.conv.weight`), so telling them apart would mean
    comparing tensor shapes. In the folder this is actually pointed at,
    `DATA/ALI/models/`, that is 244 files and 12 GB to read to find 6.6 MB.

    So it matches the token every published crown checkpoint carries. `_` and
    `-` are folded together because the two spellings are both in the wild, and
    neither ALI bundle can collide with it: theirs are `Upper_O_model.pth`,
    `Lower_MG_v6.pth` and `<landmark>_Net_<scale>.pth`.

    A checkpoint named by some other rule is not recognised, and that is the
    known cost: `find_checkpoint` then says what it looked for, and pointing
    `model` straight at the `.pth` still works.
    """
    if not name.lower().endswith(".pth"):
        return False
    return CROWN_CHECKPOINT_MARKER in name.lower().replace("_", "-")


def find_checkpoint(model_path: str) -> str:
    """The checkpoint `model` names -- given the file itself, or a folder of them.

    `model` used to be a single `.pth` and nothing else. It still is when a
    caller passes one, which is what keeps a direct call and every existing
    request working unchanged. What is new is the DIRECTORY: the server hands a
    hosted-model argument the whole of `DATA/<tool>/models/` when the caller
    named no bundle, and a tool reaching this one through the supervisor passes
    on the directory IT was handed -- ALI_IOS segmenting an unlabelled mesh
    mid-run is exactly that. Neither knows where the file sits inside, so this
    finds it.

    Walked recursively, so a checkpoint filed under a bundle folder of its own
    is found as readily as one sitting at the top.
    """
    model_path = os.fspath(model_path)
    if os.path.isfile(model_path):
        return model_path
    if not os.path.isdir(model_path):
        raise ToolInputError(f"Crown-segmentation checkpoint not found: {model_path}")

    # Relative to the folder, never absolute: these messages reach the client
    # verbatim and the server's own paths are not its business.
    candidates = sorted(
        os.path.relpath(os.path.join(root, name), model_path)
        for root, _dirs, files in os.walk(model_path)
        for name in files
        if _is_crown_checkpoint(name)
    )
    bundle = os.path.basename(model_path.rstrip(os.sep))

    if len(candidates) == 1:
        return os.path.join(model_path, candidates[0])
    if candidates:
        # Two vintages under one folder. Picking here -- the first, the newest,
        # the largest -- would leave which weights ran unrecorded, and which
        # model vintage ran must never be a surprise.
        raise ToolInputError(
            "'{}' holds {} crown-segmentation checkpoints ({}). Point 'model' at "
            "the one to use -- choosing here would leave which weights ran "
            "unrecorded.".format(bundle, len(candidates), ", ".join(candidates))
        )
    raise ToolInputError(
        "'{}' holds no crown-segmentation checkpoint: a .pth whose name carries "
        "the '{}' token every published one is named with, e.g. '{}'. Fetch it "
        "with `setup-models.sh` -- the manifest stages it under every tool that "
        "needs it -- or point 'model' straight at the .pth file.".format(
            bundle, CROWN_CHECKPOINT_MARKER, PUBLISHED_CHECKPOINT)
    )


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def is_segmented(mesh_path: str) -> bool:
    """True when the mesh already carries a per-point tooth-label array.

    Reads only the point-data array NAMES, never their values: this decides
    whether to spend minutes of GPU time, and nothing about the patient.
    """
    import vtk

    extension = os.path.splitext(mesh_path)[1].lower()
    if extension == ".stl":
        # An STL has no point data at all by construction, so it can never be
        # segmented. Reading it anyway keeps one code path.
        reader = vtk.vtkSTLReader()
    elif extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        return False

    reader.SetFileName(mesh_path)
    reader.Update()
    point_data = reader.GetOutput().GetPointData()
    names = {point_data.GetArrayName(index) for index in range(point_data.GetNumberOfArrays())}
    return bool(names & set(LABEL_ARRAY_NAMES))


def discover_meshes(input_path: str) -> list:
    """Resolve the `meshes` argument into a list of surface files.

    One mesh, or a folder of them for a batch. Folder scanning is recursive,
    and the tree is what the output mirrors.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(SURFACE_EXTENSIONS):
            raise ToolInputError(
                f"'{os.path.basename(input_path)}' is not a surface mesh. Expected one of: "
                f"{', '.join(SURFACE_EXTENSIONS)}."
            )
        return [input_path]

    found = [
        os.path.join(root, name)
        for root, _dirs, files in os.walk(input_path)
        for name in sorted(files)
        if name.lower().endswith(SURFACE_EXTENSIONS)
    ]
    if not found:
        raise ToolInputError(
            f"No surface mesh found in the input. Supported extensions: "
            f"{', '.join(SURFACE_EXTENSIONS)}."
        )
    return sorted(found)


def _input_root(input_path: str, meshes: list) -> str:
    """The directory output paths are made relative to, so a batch keeps its
    tree instead of being flattened into one folder where two patients named
    scan.vtk overwrite each other."""
    if os.path.isdir(input_path):
        return input_path
    return os.path.dirname(meshes[0])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _run_engine(csv_path: str, output_dir: str, model_path: str, input_root: str,
                array_name: str, suffix: str, device: str, fdi: bool,
                num_workers: int = 2) -> None:
    """Segment every mesh of the csv, writing each where `_predicted_path` says.

    The checkpoint is loaded once for the batch, hash-checked and read with
    `weights_only=True` (see `engine.load_network`), and the 42 cameras are
    built once. `num_workers` is accepted and unused: it sized shapeaxi's
    DataLoader, the engine reads each mesh in this process, and the argument is
    published, so a request that sends it must keep working.

    stdout is swallowed on purpose. A library under the engine may print the
    path of the scan it reads, and that path carries the patient's own file
    name, which must not reach a log. The exception it raises on a real failure
    is untouched.
    """
    engine = _import_engine()

    with open(csv_path, encoding="utf-8", newline="") as handle:
        meshes = [line.strip() for line in handle.read().splitlines()[1:] if line.strip()]
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]

    with contextlib.redirect_stdout(io.StringIO()):
        network = engine.load_network(model_path)
        network.to(device).eval()
        rig = engine.camera_rig()
        for mesh in meshes:
            engine.segment_mesh(
                network, rig, mesh,
                _predicted_path(output_dir, csv_stem, suffix, mesh, input_root),
                array_name, fdi=fdi, device=device,
            )


# The longest cause quoted back out of an engine failure. The server cuts a
# failure's reason at about 300 characters, so the cause has to fit well inside
# that together with the count that precedes it.
MAX_CAUSE = 200


def _last_meaningful_line(message) -> str:
    """The line of an exception message that actually says what went wrong.

    shapeaxi ran its meshes through a PyTorch DataLoader, and an error raised
    in a loader worker came back re-wrapped: "Caught ValueError in DataLoader
    worker process 0.", then "Original Traceback (most recent call last):",
    then the whole worker traceback, and only on the LAST line the real
    "ValueError: <message>". The server keeps the first few hundred characters
    of a reason, which would be the wrapper and the frame list and never the
    cause. The last non-blank line is the cause for that shape and is the whole
    message for an ordinary one-line error, so it is right for both.
    """
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    return last if len(last) <= MAX_CAUSE else last[:MAX_CAUSE - 3] + "..."


def _describe_failure(exc) -> str:
    """`Type: cause`, with the cause reduced to its meaningful last line.

    When the last line already names an exception type -- the DataLoader case,
    where `exc` itself is only the RuntimeError wrapper -- that inner type is
    the informative one and is not prefixed a second time.
    """
    cause = _last_meaningful_line(exc)
    head = cause.split(":", 1)[0]
    if cause and head.isidentifier() and head[:1].isupper() and (
        head.endswith("Error") or head.endswith("Exception")
    ):
        return cause
    return f"{type(exc).__name__}: {cause}" if cause else type(exc).__name__


@contextlib.contextmanager
def _counting_outputs(expected, what, interval=None):
    """Report "mesh k of n" while the engine runs, from the outputs that exist.

    The engine takes the whole batch in one call and says nothing on the way, but
    it writes each mesh's segmented copy as soon as that mesh is done, to a
    path known in advance. So the number of those files that exist IS the
    number of meshes finished -- a count, not an estimate -- and a thread that
    looks at them every few seconds can move the bar truthfully.

    A file that was already there before the call is not counted: it says
    nothing about this run, and counting it would claim a mesh still to come.
    The thread is a daemon, is stopped in a `finally` whatever the call does,
    and swallows every error of its own -- progress must never fail a run.
    """
    interval = WATCH_INTERVAL if interval is None else interval
    total = len(expected)
    try:
        fresh = [path for path in expected if not os.path.exists(path)]
    except Exception:  # noqa: BLE001 -- telemetry must never fail a run
        fresh = []
    stop = threading.Event()

    def watch():
        reported = 0
        while not stop.wait(interval):
            try:
                done = sum(1 for path in fresh if os.path.exists(path))
                # The last mesh is not reported here: once it exists the call
                # is about to return, and the run's own next step says so.
                if reported < done < total:
                    reported = done
                    progress.report(done + 1, total, what)
            except Exception:  # noqa: BLE001 -- telemetry must never fail a run
                pass

    thread = threading.Thread(target=watch, name="crownseg-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval + 1.0)


def _predicted_path(output_dir: str, csv_stem: str, suffix: str, mesh: str,
                    input_root: str) -> str:
    """Where the engine writes the segmented copy of `mesh`.

    The layout is shapeaxi's csv branch (`save_data_vtk_from_csv`), kept so the
    output tree is the one every consumer already reads: the input path minus its extension, plus
    `_<suffix>.vtk`, with `vtk_folder` stripped off the front, filed under
    `<out>/<csv stem>_<suffix>/`. Recomputed rather than parsed back out of
    shapeaxi's stdout, which is where the file names this tool must not log
    would have had to travel.
    """
    without_extension = os.path.splitext(mesh)[0] + f"_{suffix}.vtk"
    relative = without_extension.replace(input_root, "").lstrip(os.sep)
    return os.path.normpath(os.path.join(output_dir, f"{csv_stem}_{suffix}", relative))


def segment_crowns(
    input_path: str,
    model_path: str,
    output_dir: str,
    array_name: str = DEFAULT_ARRAY_NAME,
    suffix: str = DEFAULT_SUFFIX,
    fdi: bool = False,
    skip_segmented: bool = True,
    device: str = "cuda",
    num_workers: int = 2,
) -> dict:
    """Label the teeth of one mesh or a whole folder of them.

    `skip_segmented=True` passes an already-labelled mesh through unchanged
    instead of re-running the network on it, which is what makes a mixed batch
    of raw and pre-segmented meshes one call.
    """
    started_at = time.monotonic()

    array_name = (array_name or DEFAULT_ARRAY_NAME).strip() or DEFAULT_ARRAY_NAME
    suffix = (suffix or DEFAULT_SUFFIX).strip() or DEFAULT_SUFFIX

    output_dir = os.fspath(output_dir)

    # `model` may be the checkpoint itself or a folder to find it in; see
    # find_checkpoint. Resolved HERE, before the scratch directory exists: a
    # request the tool refuses must leave the caller's output directory exactly
    # as it found it, and a 422 that has already written a folder into the
    # response's own directory is a refusal that made a mess.
    checkpoint = find_checkpoint(model_path)

    meshes = discover_meshes(os.fspath(input_path))
    input_root = _input_root(os.fspath(input_path), meshes)

    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)
    try:
        return _segment(
            meshes=meshes,
            input_root=input_root,
            model_path=checkpoint,
            output_dir=output_dir,
            work_dir=work_dir,
            array_name=array_name,
            suffix=suffix,
            fdi=fdi,
            skip_segmented=skip_segmented,
            device=device,
            num_workers=num_workers,
            started_at=started_at,
        )
    finally:
        # `finally`, not "after the run and again in an except": the scratch
        # directory used to survive every error path that was not the engine's
        # own, and it holds a csv listing the patient's file paths.
        shutil.rmtree(work_dir, ignore_errors=True)


def _segment(meshes, input_root, model_path, output_dir, work_dir, array_name, suffix,
             fdi, skip_segmented, device, num_workers, started_at) -> dict:
    """The run itself, once the arguments are known to be usable.

    Split out so `segment_crowns` can own the scratch directory's lifetime in
    one `try/finally` instead of repeating the cleanup on each way out.
    """
    already_segmented, to_segment = [], []
    for mesh in meshes:
        (already_segmented if skip_segmented and is_segmented(mesh) else to_segment).append(mesh)

    # Probed HERE, before the split decides whether the engine is ever reached.
    # The engine is imported inside `_run_engine`, which only runs when
    # something needs segmenting -- so a batch that happens to be entirely
    # pre-labelled returned a clean report on a deployment where the
    # `segmentation` extra is not installed at all, and the same deployment
    # failed the moment one raw mesh appeared. A tool that can serve some
    # batches and not others is not available; it just has not been asked the
    # right question yet.
    #
    # Not raised here: a pre-labelled batch is still legitimately served, and
    # refusing it would break the Crown_Seg -> ALI chain for callers who
    # segmented elsewhere. It is RECORDED, so the report says what this run
    # could and could not have done. The real fix is server-side -- the
    # registry must refuse to serve a tool whose declared extras are absent,
    # the same way it refuses a tool whose schema no longer matches its source.
    try:
        _import_engine()
        engine_available, engine_error = True, None
    except Exception as exc:  # ToolUnavailableError, or anything its import raises
        engine_available, engine_error = False, f"{type(exc).__name__}: {exc}"
        # The operator's to fix, not the clinician's: the deployment is missing
        # an extra. The reason itself stays in the report, where an install
        # path in it reaches nobody but the caller.
        progress.log(
            "the segmentation engine is not installed here; only meshes that "
            "already carry labels can be served", "warning",
        )

    records = {}
    produced = []

    # A mesh that already carries labels is copied, not re-predicted. Copied
    # rather than referenced in place because the caller is handed one output
    # tree, and because the input may live on a read-only mount.
    for mesh in already_segmented:
        relative = os.path.relpath(mesh, input_root)
        base = os.path.splitext(os.path.basename(mesh))[0]
        destination = os.path.join(
            output_dir, os.path.dirname(relative), f"{base}_{suffix}.vtk"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _write_as_vtk(mesh, destination)
        # `engine_unavailable` when this deployment could not have segmented
        # anything anyway. The name is the server's existing vocabulary for
        # "this installation cannot do that" -- ToolUnavailableError, answered
        # as 501 -- so the status reads with it rather than beside it.
        # `already_segmented` means "it did not need doing";
        # `engine_unavailable` means "it did not need doing, AND could not have
        # been done". `engine_error` holds the reason.
        #
        # Putting it in the status rather than only in a field further down is
        # the right PLACE, not a guarantee it is seen. `ALI_IOS` DOES read
        # this report -- it is how it tells a mesh that was labelled from one
        # that could not be -- so the vocabulary here is a contract, not a
        # note for a human.
        records[relative] = {
            "status": "already_segmented" if engine_available else "engine_unavailable",
            "input": relative,
            "produced": [destination],
        }
        produced.append(destination)

    if to_segment:
        device = resolve_device(device)

        # Written into the work dir. The Slicer module wrote this csv into the
        # extension's own source folder, which breaks a read-only install and
        # leaves a file behind pointing at the patient's data.
        csv_path = os.path.join(work_dir, "crownseg_input.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("surf\n")
            for mesh in to_segment:
                handle.write(f"{mesh}\n")

        total = len(to_segment)
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        expected = [
            _predicted_path(output_dir, csv_stem, suffix, mesh, input_root)
            for mesh in to_segment
        ]
        # Said BEFORE the call, because a failure inside the engine arrives as one
        # opaque exception for the whole batch: this line is what tells the
        # operator which checkpoint, how many meshes and which device were
        # being attempted when it came. The checkpoint's name is the model's,
        # never a patient's.
        logger.info(
            "loading checkpoint %s and segmenting %d mesh(es) on %s",
            os.path.basename(model_path), total, device,
        )
        # The model load inside the engine is not visible from here, so the first
        # mesh is announced at the start of the bar and the load is part of it.
        progress.report(1, total, "mesh")
        try:
            with _counting_outputs(expected, "mesh"):
                _run_engine(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    model_path=model_path,
                    input_root=input_root,
                    array_name=array_name,
                    suffix=suffix,
                    device=device,
                    fdi=fdi,
                    num_workers=num_workers,
                )
        except (ToolUnavailableError, ToolInputError):
            # Already phrased for whoever has to act on them, and classed so the
            # server answers 503 or 422; rewrapping would turn both into a 500.
            raise
        except Exception as exc:
            # the engine takes the batch in one call, so its failure carries no
            # position. The outputs on disk do: they say how far it got.
            written = sum(1 for path in expected if os.path.isfile(path))
            cause = _describe_failure(exc)
            logger.error(
                "the engine failed after %d of %d meshes were written (%s)",
                written, total, cause,
            )
            # RuntimeError whatever the engine raised: a ValueError out of its
            # loader is a fault of the bundled engine or of this server, not
            # an argument the caller sent, and must not be answered as a 422.
            # The cause comes first so the server's cut keeps it.
            raise RuntimeError(
                f"crown segmentation failed: {cause} "
                f"({written} of {total} meshes were written before it)"
            ) from exc

        for index, mesh in enumerate(to_segment, start=1):
            relative = os.path.relpath(mesh, input_root)
            predicted = _predicted_path(output_dir, csv_stem, suffix, mesh, input_root)
            if os.path.isfile(predicted):
                records[relative] = {"status": "segmented", "input": relative,
                                     "produced": [predicted]}
                produced.append(predicted)
            else:
                # One mesh the engine could not write must not cost the batch.
                logger.warning(
                    "mesh %d of %d: segmentation failed (the engine returned "
                    "without error but wrote no output for it)", index, total,
                )
                progress.log(
                    f"mesh {index} of {total} produced no segmentation",
                    "warning", user=True,
                )
                records[relative] = {
                    "status": "failed",
                    "input": relative,
                    "produced": [],
                    "error": "the segmentation produced no output for this mesh",
                }

    failed = sum(1 for r in records.values() if r["status"] == "failed")
    if not produced:
        # Only reachable when every mesh went to the engine and it returned
        # cleanly having written none of them -- an exception would have been
        # raised above. Saying so explicitly is the point: "no output" with no
        # error is a different fault (a wrong output layout, a silent skip)
        # from a crash, and the operator has nothing else to tell them apart.
        raise RuntimeError(
            f"0 of {len(meshes)} meshes segmented; the engine returned without "
            f"error but wrote none of the {len(to_segment)} expected outputs"
        )

    labelled = len(produced)
    logger.log(
        logging.WARNING if failed else logging.INFO,
        "%d of %d meshes labelled, %d failed (%d segmented here, %d already labelled)",
        labelled, len(meshes), failed, labelled - len(already_segmented),
        len(already_segmented),
    )

    report = {
        "tool": TOOL_NAME,
        # WHICH weights ran, by name. `model` may now be a folder holding
        # several, so the report has to say which one was picked out of it --
        # the argument no longer answers that on its own.
        "checkpoint": os.path.basename(model_path),
        "array_name": array_name,
        "suffix": suffix,
        "numbering": "FDI" if fdi else "Universal",
        "device": device if to_segment else None,
        "cases": records,
        # Absolute paths of every mesh that now carries tooth labels, whether
        # this run produced them or found them already labelled. This is what a
        # caller sequencing CrownSeg before ALI reads.
        "segmented_meshes": produced,
        # So a caller can tell "nothing needed segmenting" from "nothing COULD
        # have been segmented". Without it the two produce an identical report.
        "engine_available": engine_available,
        "engine_error": engine_error,
        "summary": {
            "total": len(meshes),
            "segmented": sum(1 for r in records.values() if r["status"] == "segmented"),
            # Counted off the records rather than off `already_segmented`, so
            # these two never contradict the per-mesh statuses above: a
            # degraded run reports 0 already_segmented and N
            # engine_unavailable, not N of both.
            "already_segmented": sum(
                1 for r in records.values() if r["status"] == "already_segmented"
            ),
            "engine_unavailable": sum(1 for r in records.values() if r["status"] == "engine_unavailable"),
            "failed": sum(1 for r in records.values() if r["status"] == "failed"),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }

    with open(os.path.join(output_dir, "run_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def _write_as_vtk(source: str, destination: str) -> None:
    """Copy a mesh to `destination` as .vtk polydata.

    A pre-segmented input can be an .stl, and the outputs are always
    .vtk; normalizing here means the caller gets one format whichever branch a
    mesh took, instead of an .stl that every downstream reader then has to
    special-case. (The Slicer module copied it verbatim, and its landmark CLI
    then discovered only .vtk -- so those meshes were silently never used.)
    """
    import vtk

    extension = os.path.splitext(source)[1].lower()
    reader = vtk.vtkSTLReader() if extension == ".stl" else vtk.vtkPolyDataReader()
    reader.SetFileName(source)
    reader.Update()

    writer = vtk.vtkPolyDataWriter()
    # Slicer reads a .vtk without a SPACE= header as LPS and warns about it
    # on every load; saying LPS keeps exactly that reading, without the warning.
    writer.SetHeader("SPACE=LPS")
    writer.SetFileName(destination)
    writer.SetInputData(reader.GetOutput())
    writer.Write()

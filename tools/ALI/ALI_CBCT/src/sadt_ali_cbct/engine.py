"""The CBCT landmark pipeline: preprocess, then one agent per landmark.

Ported from ALI_CBCT/ALI_CBCT.py. What the CLI envelope did -- argparse,
`ast.literal_eval` on stringified lists, `sys.exit()`, the
`<filter-progress>` protocol -- is gone, and a failure has to be an exception
rather than a `SystemExit` the caller cannot catch cleanly.

Beyond that, two behaviours changed on purpose:

* **Weights are discovered, landmarks are requested.** The bundle's folder
  tree says which landmarks it can predict; the caller's region selection says
  which are wanted. What is wanted but absent is reported, not silently
  skipped.
* **One markups file per scan**, holding every landmark found. The original
  wrote one file per anatomical region, so every downstream tool (ASO, AREG,
  AutoMatrix) had to recombine them by hand.

The GPU semaphore this carried is gone with the shared server process: a tool
is now its own process, so an in-process limit would cap nothing. Capping GPU
work across concurrent jobs is the server's, and it has to be across tools.
"""

from concurrent import futures
import logging
import multiprocessing
import os
import shutil
import time

import numpy as np

from .errors import ToolInputError
from sadt_ali_common.markups import MARKUPS_EXTENSION
from sadt_ali_common.markups import write as write_markups
from . import catalog
from . import preprocess
from . import progress
from .agent import AGENT_FOV, MOVEMENT_COUNT, Agent, NotFound, rng_for
from .brain import Brain, import_torch, resolve_device

logger = logging.getLogger(__name__)

# Per-landmark search budget when `search_steps` is left at 0, counted in
# FORWARD PASSES rather than in seconds.
#
# 900 is what the 15 s this used to allow actually bought on this card:
# measured at 59 steps/s, while the worst SUCCESSFUL search on the reference
# scan takes 193 steps (walk 158 + refine 35; mean 132). So the bound is 4.7x
# the worst real search and lands where the old one did -- a landmark that was
# found is still found, one that was not still gives up.
#
# It is one number for every device, where the old one had to be four times
# larger on the CPU to buy the same search. A step count is a property of the
# SEARCH; seconds were a property of the machine, and of how busy it was.
_DEFAULT_SEARCH_STEPS = 900

# The widest a run asks for, however much the machine could afford.
#
# Measured on the reference scan's 119 landmarks, card otherwise idle, every
# width placing the same 114 points at the same coordinates:
#
#   width 1   358.7 s      width 8   128.0 s   (x2.80)
#   width 4   141.3 s      width 12  127.3 s   (x2.82)
#
# **Eight is where the gain stops, not where the tool breaks.** Twelve buys
# 0.7 s for three more gigabytes of card, and a run RESERVES what it is
# granted -- so asking wider takes room from other runs and gives this one
# nothing. Past twelve nothing is measured at all.
#
# It is a ceiling on the ASK, never a floor: the supervisor still answers with
# less whenever the machine is busy, and one landmark still opens one channel.
# A number inside a tool cannot see the machine, which is why this is the
# largest width MEASURED to be worth having rather than a guess at what the
# hardware can take -- the distinction this repository already draws after
# CLIC opened thirty-three channels for a cohort of six.
MAX_AGENT_CHANNELS = 8

COMPOUND_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz")


def search_budget(search_steps: int = 0) -> int:
    """The per-landmark step budget: the caller's, or the default above.

    0 means "not specified" -- there is no nullable type in the schema, so the
    argument cannot default to None the way the setting it replaces did.

    It takes no device any more, and that is the whole point of counting steps:
    the same number means the same search on a laptop and on this card.
    """
    if search_steps and int(search_steps) > 0:
        return int(search_steps)
    return _DEFAULT_SEARCH_STEPS


# numpy's legacy global seeder is the narrowest of the streams pinned below and
# fixes the accepted range: torch would take a wider one, `default_rng` any
# non-negative integer at all.
_MAX_SEED = 2 ** 32 - 1


def seed_everything(seed: int) -> None:
    """Pin every global random stream this run could reach.

    What actually makes a prediction reproducible is that each agent draws from
    its own generator (`agent.rng_for`); this is the belt to that pair of
    braces. `random`, `numpy` and torch's CPU and CUDA streams are pinned so
    that nothing called along the way -- a monai transform, a later change in
    this file -- can reintroduce an unseeded draw without someone having to do
    it deliberately.

    No DataLoader worker seeding is needed and none is done, because there is
    no DataLoader: `Environment` composes BorderPad, EnsureChannelFirst,
    ScaleIntensity and SpatialCrop, none of which is a `Rand*` transform, and
    every forward pass runs in the calling thread. Adding a loader with workers
    later would add a stream this function does not cover.

    cuDNN's algorithm choice is left alone. It is not pinned here because it
    was measured not to matter: across six runs of the reference scan, 113 of
    the 114 landmarks were bit-identical, so the forward passes already agree.
    Forcing deterministic kernels would cost speed to fix something that is not
    broken, and would hide it if it ever did break.
    """
    import random

    torch = import_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_dependencies() -> None:
    """Import the whole lazy stack once, before any scan is touched.

    A missing dependency is a property of the VENV, not of one scan. Without
    this, the per-scan `except` below catches it as if a single patient's data
    were at fault: every scan fails identically -- each only after a complete
    histogram correction, so a 200-scan cohort spends real minutes discovering
    the same thing 200 times -- and the run then ends on "ALI produced no
    landmarks for any scan", which buries the one line that says what to
    install.

    Raising here instead surfaces the install message itself, immediately.
    """
    from .environment import import_transforms

    import_torch()
    preprocess.import_itk()
    import_transforms()


def scan_stem(filename: str) -> str:
    """'patient01.nii.gz' -> 'patient01', compound extensions preserved."""
    lower = filename.lower()
    for suffix in COMPOUND_EXTENSIONS:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return os.path.splitext(filename)[0]


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def discover_weights(model_path: str) -> dict:
    """{landmark: {scale key: checkpoint path}} for a CBCT model bundle.

    The layout is the one every published packaging of these weights uses:
    `<...>/<landmark>/<scale>/<anything>.pth`, where the scale folder is named
    after the spacing (`1`, `0-3`). Walked recursively, so it does not matter
    how deeply the region folders nest the landmarks -- which is what lets the
    eight separately published region archives be unpacked side by side into
    a single bundle.

    Aliased landmark spellings are folded onto the canonical one, so a bundle
    naming the impacted canines the way the old Slicer UI did resolves to the
    same landmark as one naming them the way the CLI did.
    """
    if not os.path.isdir(model_path):
        # An argument error, not a fault of this tool: the caller pointed
        # `model` at something that is not a CBCT bundle. Basename only -- the
        # message reaches the client verbatim, the server's paths do not.
        raise ToolInputError(
            f"CBCT model bundle '{os.path.basename(model_path.rstrip(os.sep))}' "
            f"is not a directory."
        )

    weights: dict = {}
    # Where each (landmark, scale) was found. The walk is recursive on purpose --
    # it is what lets the eight published region archives be unpacked side by
    # side -- and those contribute DIFFERENT landmarks, so they never collide.
    # Two whole bundles under one folder do, and the plain assignment below used
    # to let the second silently overwrite the first, landmark by landmark: the
    # result was a mixture of two vintages that no report could name. Which model
    # vintage ran must never be a surprise.
    seen: dict = {}
    for root, _dirs, files in os.walk(model_path):
        checkpoints = [name for name in sorted(files) if name.endswith(".pth")]
        scale = os.path.basename(root)
        if not checkpoints or scale not in catalog.SCALE_KEYS:
            continue
        label = catalog.canonical(os.path.basename(os.path.dirname(root)))
        where = os.path.relpath(root, model_path)
        if (label, scale) in seen:
            # Relative to the bundle, never absolute: this message reaches the
            # client verbatim and the server's own paths are not its business.
            raise ToolInputError(
                "This model folder holds more than one bundle: '{}' at scale '{}' "
                "is in both '{}' and '{}'. Point 'model' at one of them -- "
                "choosing here would leave which weights ran unrecorded.".format(
                    label, scale, seen[(label, scale)], where
                )
            )
        seen[(label, scale)] = where
        weights.setdefault(label, {})[scale] = os.path.join(root, checkpoints[0])

    # A landmark needs a checkpoint at EVERY scale: the agent walks the coarse
    # one and then the fine one. Filtering here means a half-copied bundle is
    # reported up front rather than failing in the middle of the run.
    return {
        label: scales
        for label, scales in weights.items()
        if all(scale in scales for scale in catalog.SCALE_KEYS)
    }


def requested_landmarks(weights: dict, regions, landmarks=()):
    """(runnable, without_model, ungrouped) for a bundle and a selection.

    "Requested" is every catalog landmark of the selected regions, plus any
    landmark the bundle provides whose region is selected -- so weights for a
    landmark this catalog has not heard of surface as `ungrouped` rather than
    being silently ignored.

    An explicit `landmarks` list REPLACES the regions rather than narrowing
    them, which is what lets a caller ask for exactly the points it needs: ASO
    registers on seven landmarks straddling two regions, so going through
    `regions` would run 58 agents to use 7. Narrowing would agree here only
    because ASO leaves the regions all on, and would silently drop landmarks
    for any caller that set both.

    Empty (the default) means "not specified" and leaves the regions in charge.
    """
    regions = set(regions)
    ungrouped = []

    if landmarks:
        wanted = set(landmarks)
        # An explicitly named landmark the catalog does not know is still run
        # if the bundle has weights for it: the caller named it on purpose.
        ungrouped = sorted(
            label for label in wanted if catalog.group_of(label) == catalog.UNGROUPED
        )
        runnable = tuple(sorted(label for label in wanted if label in weights))
        without_model = sorted(label for label in wanted if label not in weights)
        return runnable, without_model, ungrouped

    wanted = set(catalog.landmarks_in(regions))

    for label in weights:
        group = catalog.group_of(label)
        if group == catalog.UNGROUPED:
            ungrouped.append(label)
        elif group in regions:
            wanted.add(label)

    runnable = tuple(sorted(label for label in wanted if label in weights))
    without_model = sorted(label for label in wanted if label not in weights)
    return runnable, without_model, sorted(ungrouped)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _prepare_scan(scan_path: str, work_dir: str) -> dict:
    """Histogram-correct one scan and resample it to every scale.

    Returns {scale key: path}, written under `work_dir` -- always the request's
    own scratch directory, never next to the input, which is either read-only
    server-side data or an extracted upload.
    """
    os.makedirs(work_dir, exist_ok=True)

    base = os.path.basename(scan_path)
    corrected = os.path.join(work_dir, base)
    preprocess.correct_histogram(scan_path, corrected)

    stem = scan_stem(base)
    resampled = {}
    for spacing in catalog.SCALE_SPACINGS:
        key = catalog.scale_key(spacing)
        # Written as NIfTI whatever the input was: this is a real read/write
        # conversion, not the rename the original relied on for NRRD and GIPL.
        destination = os.path.join(work_dir, f"{stem}_sp{key}.nii.gz")
        preprocess.set_spacing(corrected, spacing, destination)
        resampled[key] = destination
    return resampled


def _channels_for(sup, wanted: int, declared: int = 0) -> int:
    """How many agents to walk at once: what the machine will pay for.

    The tool asks for its own count -- the landmarks it was told to place --
    and the supervisor answers with what this run's reserved share can afford,
    floor one. A hundred and nineteen agents is never what comes back: one
    holds its own networks on the card, so the budget clamps the answer long
    before the ask does.

    **`declared` is `num_workers`, and declaring it is what turns any of this
    on.** The server's `execution/concurrency` is inert for a tool whose schema
    names neither `num_workers` nor `batch_size`: it reserves no room, sets no
    `SADT_CHANNEL_BUDGET`, and `sup.channels()` then answers 1 before doing any
    arithmetic at all. A number the caller named is a ceiling on the ask, never
    a floor over it -- admission reserved against what it granted.

    One without a supervisor unless the caller named a number, which is how
    this tool is run from a CLI and from its own tests: nothing has reserved
    anything, and opening a hundred networks on an unknown card is a way to be
    killed rather than a way to be fast.
    """
    wanted = max(1, min(int(wanted), MAX_AGENT_CHANNELS))
    try:
        declared = int(declared or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > 0:
        wanted = min(wanted, declared)

    ask = getattr(sup, "channels", None)
    if ask is None:
        return max(1, wanted) if declared > 0 else 1
    try:
        return max(1, min(wanted, int(ask(wanted))))
    except Exception:  # noqa: BLE001 - a grant must never fail a run
        logger.warning("Could not ask for channels; searching one at a time",
                       exc_info=True)
        return 1


def _walk_on_one_thread(device: str) -> None:
    """One CPU thread for the process that walks, and this is what makes the
    width pay.

    The volume lives in host memory, so every step crops and rescales it on
    the CPU -- 0.9 ms of arithmetic that enters torch's intra-op pool and
    occupies about ten cores while it does. Multiplied by the width, that is
    more threads than the machine has cores, and the walkers spend their time
    taking the card away from each other: measured on the full 119 landmarks,
    width 8 came out at 425 s against 359 s for width 1, and 128.0 s once each
    worker was held to one thread.

    Called in the PARENT as well as in each worker, because the parent is what
    walks at width 1 -- and because the server reserves against the cores a run
    was SEEN to occupy. A run measured at 2.2 cores a channel is priced at 2.9
    with the safety margin, so on 42 cores the ladder stops at fourteen
    channels: the occupancy is not just waste, it is the bound on the width.

    Only on the card. A CPU deployment does its FORWARD pass in this pool too
    -- 31.6 ms on 28 threads against 143.9 ms on one -- so capping it there
    would cost such a run 4.6x for nothing.
    """
    if not device.startswith("cuda"):
        return
    try:
        import_torch().set_num_threads(1)
    except Exception:  # noqa: BLE001 - a thread cap must never fail a run
        logger.warning("Could not cap this process's torch threads", exc_info=True)


def _agent_padding():
    """Half a field of view plus one, so a box centred anywhere inside the
    volume is still complete once the borders are padded."""
    return np.array(AGENT_FOV) / 2 + 1


def _walk_one(label, environment, weights, device, budget, seed):
    """One agent's whole search, and the ONLY implementation of it.

    Both the serial path and the pool call this, so a run at any width
    executes the same lines in the same order -- which is what makes "the
    width does not move a coordinate" a property of the code rather than a
    claim about it.
    """
    brain = Brain(catalog.SCALE_KEYS, device, out_channels=MOVEMENT_COUNT)
    try:
        brain.load(weights[label])
        agent = Agent(
            target=label,
            scale_keys=catalog.SCALE_KEYS,
            brain=brain,
            environment=environment,
            # Per landmark, not per scan: see agent.rng_for. This is what lets
            # the agents run in any order, or at once, without the answer
            # depending on which OTHER landmarks were asked for.
            rng=rng_for(label, seed),
        )
        return agent.search(budget), None
    except NotFound as exc:
        # At INFO, not DEBUG: these are rare (6 of 119 on the reference scan)
        # and they are exactly what someone watching the log wants to see,
        # without having to open the archive to find out.
        logger.info("  %s: not found -- %s", label, exc)
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - one landmark, not the scan
        # A broken checkpoint, an out-of-memory, anything: this landmark is
        # lost, the rest of the scan is not.
        logger.exception("Landmark search raised for '%s'", label)
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        brain.release()


# **The agents walk in separate PROCESSES, and threads are not an oversight.**
#
# They were measured twice and refused twice. A thread pool over the same
# agents is x1.04 on this card, because a step is 15.5 ms of which 15.2 is the
# forward pass and essentially all of that is spent holding the GIL inside
# torch's dispatch -- the C++ that actually releases it is a few hundred
# microseconds of kernel. There is nothing for a second thread to overlap
# with. Processes have no such ceiling: 353.8 s at width 1, 210.1 s at width 2,
# 169.1 s at width 4, measured on this engine.
#
# The price is real and is paid once per worker: a spawned child builds its own
# CUDA context (~3 s) and loads its own copy of the scan. So the pool is
# created ONCE for the run, not once per scan, and a worker keeps the
# environment it built until the scan changes under it.
_WORKER = {}


def _worker_setup(device, padding, weights, budget, seed) -> None:
    """Called once in each spawned child, before any landmark reaches it."""
    _WORKER.update(
        device=device, padding=padding, weights=weights, budget=budget,
        seed=seed, images=None, environment=None,
    )
    # The walk draws only from `agent.rng_for`, so this seeds nothing the
    # search depends on. It is set anyway because a child that imports torch
    # inherits no global state at all under `spawn`, and an unseeded one would
    # be the single hardest thing to notice if that ever stopped being true.
    seed_everything(seed)

    _walk_on_one_thread(device)


def _worker_environment(images, key):
    """This process's volumes for one scan, loaded once and then reused.

    Keyed on `images` -- the {scale: path} mapping -- rather than on the scan
    index, so a worker that never saw a scan loads it and one that already has
    it does not. Across a cohort each worker pays one load per scan, never one
    per landmark.
    """
    if _WORKER.get("images") != images:
        _worker_release()
        from .environment import Environment

        environment = Environment(
            patient_id=key, padding=_WORKER["padding"], device=_WORKER["device"],
        )
        environment.load_images(images)
        _WORKER["environment"] = environment
        _WORKER["images"] = images
    return _WORKER["environment"]


def _worker_release() -> None:
    environment = _WORKER.get("environment")
    if environment is not None:
        environment.release()
    _WORKER["environment"] = None
    _WORKER["images"] = None


def _worker_walk(task):
    """One task: place one landmark on one scan, in this child."""
    images, key, label = task
    environment = _worker_environment(images, key)
    return _walk_one(
        label, environment, _WORKER["weights"], _WORKER["device"],
        _WORKER["budget"], _WORKER["seed"],
    )


def _walk_in_pool(pool, images, key, runnable, announce):
    """Hand every landmark to the workers; return what came back, and whether
    the pool survived.

    **A process pool has a failure mode a thread pool does not**, and it is the
    reason this returns a flag rather than raising. A worker that is KILLED --
    the host out of memory, an operator, the card -- breaks the executor
    permanently: every pending future fails at once and every later `submit`
    raises, so one dead worker would otherwise cost the rest of the cohort and
    not just the landmark it was holding. What comes back here is therefore
    partial by design, and the caller finishes the remainder in this process.
    """
    results = {}
    broken = False
    pending = {}
    try:
        for label in runnable:
            pending[pool.submit(_worker_walk, (images, key, label))] = label
    except Exception:  # noqa: BLE001 - a dead pool is not a failed run
        logger.exception("Could not hand the landmarks to the agent pool")
        broken = True

    for future in futures.as_completed(pending):
        label = pending[future]
        try:
            results[label] = future.result()
        except futures.BrokenExecutor:
            # Not this landmark's fault and not recorded against it: it is
            # walked again below, in this process, and only a real NotFound
            # should ever reach the run report.
            broken = True
        except Exception as exc:  # noqa: BLE001 - one landmark, not the scan
            # `_walk_one` catches everything a search itself can raise, so
            # reaching here means the call did not survive the trip.
            logger.exception("Landmark search did not come back for '%s'", label)
            results[label] = (None, f"{type(exc).__name__}: {exc}")
        announce(len(results))
    return results, broken


def predict_landmarks(
    scans: list,
    model_path: str,
    regions=None,
    landmarks=(),
    prediction_ID: str = "Pred",
    output_dir: str = None,
    work_dir: str = None,
    device: str = None,
    search_steps: int = 0,
    seed: int = 0,
    num_workers: int = 0,
    sup=None,
) -> dict:
    """Place landmarks on every scan; return the run report.

    `scans` is a list of `(absolute path, key)` pairs, where the key is the
    scan's path relative to the input root. `dispatch` owns discovery and hands
    the keys over; this owns inference. Keying by relative path rather than by
    base name is what stops two patients called `scan.nii.gz` in different
    subfolders from overwriting each other -- silently, in the original, both
    in the working dictionary and in the flat output folder.

    The loop is scan-outer, landmark-inner: one scan's volumes and one
    landmark's networks are in memory at a time. Inverting it would load each
    checkpoint once instead of once per scan, but would need either every scan
    resident at both spacings, or all 112 possible landmarks' networks resident
    on the card. Neither fits a cohort.
    """
    started_at = time.monotonic()

    check_dependencies()
    device = resolve_device(device)
    # Before anything walks, and in this process too: at width 1 the parent IS
    # the walker, and the pool it would otherwise open is what the cost table
    # then reads back as cores this run needs.
    _walk_on_one_thread(device)
    regions = tuple(regions) if regions is not None else catalog.REGION_CODES
    landmarks = tuple(landmarks or ())
    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"
    budget = search_budget(search_steps)

    seed = int(seed)
    if not 0 <= seed <= _MAX_SEED:
        # An argument error, like the ones below: Slicer shows it verbatim.
        raise ToolInputError(f"seed must be between 0 and {_MAX_SEED}; got {seed}.")
    seed_everything(seed)

    weights = discover_weights(model_path)
    if not weights:
        # An input error, not a crash, here and below: nothing the server can
        # do -- the caller must pick the bundle (or the regions) that match.
        # Slicer shows these messages verbatim.
        raise ToolInputError(
            f"No CBCT landmark weights found in '{os.path.basename(model_path)}'. Expected "
            f"<bundle>/**/<landmark>/<scale>/*.pth, with scale folders named "
            f"{' and '.join(catalog.SCALE_KEYS)}."
        )

    runnable, without_model, ungrouped = requested_landmarks(weights, regions, landmarks)
    if not runnable:
        # Name what was actually asked for. Reporting the regions when the
        # caller named landmarks would point at a selection they never made.
        if landmarks:
            asked = f"landmark(s) ({', '.join(landmarks)})"
        else:
            asked = (
                f"region(s) ("
                f"{', '.join(catalog.REGION_DISPLAY_NAMES.get(code, code) for code in regions)})"
            )
        raise ToolInputError(
            f"'{os.path.basename(model_path)}' has no weights for any of the selected "
            f"{asked}. It provides: {', '.join(sorted(weights)) or 'nothing'}."
        )

    preprocessed_dir = os.path.join(work_dir, "preprocessed")
    logger.info(
        "ALI CBCT: %d scan(s), %d landmark(s), device=%s", len(scans), len(runnable), device
    )

    # Asked once for the run, not once per scan: the grant is a property of
    # what admission reserved, and re-asking per scan would only repeat it.
    width = _channels_for(sup, len(runnable), num_workers)
    logger.info("ALI CBCT: walking %d agent(s) %d at a time", len(runnable), width)
    # Declared around the search and nowhere else: the peak is there -- one
    # agent's networks per channel -- and the preprocessing either side is
    # serial. A record without a width is ignored by the server rather than
    # read as one, which is what lets a tool speak only where it has something
    # to say.
    progress.set_width(width)

    # `spawn`, not the platform default: a forked child inherits a CUDA context
    # it cannot use, and the first allocation in it fails. Every worker builds
    # its own, which is most of what a channel costs on the card.
    pool = None
    if width > 1 and len(runnable) > 1:
        pool = futures.ProcessPoolExecutor(
            max_workers=width,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_setup,
            initargs=(device, _agent_padding(), weights, budget, seed),
        )

    scan_reports = {}
    try:
        _predict_every_scan(
            scans=scans, scan_reports=scan_reports, weights=weights,
            runnable=runnable, device=device, budget=budget,
            preprocessed_dir=preprocessed_dir, output_dir=output_dir,
            prediction_ID=prediction_ID, seed=seed, pool=pool,
        )
    finally:
        # Before the report is built, and on the failure path too: a pool left
        # open holds `width` processes, each with a CUDA context and a copy of
        # the last scan, for as long as this interpreter lives.
        if pool is not None:
            pool.shutdown(wait=True)
        progress.set_width(None)

    processed = [record for record in scan_reports.values() if record["status"] == "ok"]
    if not processed:
        first_error = next(
            (record.get("error") for record in scan_reports.values() if record.get("error")),
            "unknown",
        )
        raise RuntimeError(f"ALI produced no landmarks for any scan. First error: {first_error}")
    processed = [record for record in scan_reports.values() if record["status"] == "ok"]
    if not processed:
        first_error = next(
            (record.get("error") for record in scan_reports.values() if record.get("error")),
            "unknown",
        )
        raise RuntimeError(f"ALI produced no landmarks for any scan. First error: {first_error}")

    written = sum(len(record["landmarks_found"]) for record in scan_reports.values())
    # The guard above counts SCANS; this one counts what the tool claims to
    # have produced. They are not the same number, and the difference is the
    # whole failure mode: before the check above learned to look at
    # `landmarks_found`, a batch on which every agent failed reported every
    # scan "ok" and returned 200 with no coordinates in it. Redundant today by
    # construction, kept because the two guards answer different questions and
    # only this one is about the output.
    if not written:
        raise RuntimeError(
            f"ALI placed no landmark on any of the {len(scan_reports)} scan(s). "
            f"{len(runnable)} landmark(s) were requested and the bundle had weights "
            f"for all of them, so this is a failure of the search, not of the selection."
        )
    never_found = sorted(
        {label for record in scan_reports.values() for label in record["landmarks_failed"]}
    )
    logger.info(
        "ALI CBCT done: %d/%d scan(s), %d landmark(s) written, %.0fs",
        len(processed), len(scan_reports), written, time.monotonic() - started_at,
    )
    if without_model:
        logger.info("  not in this bundle (%d): %s", len(without_model), ", ".join(without_model))
    if never_found:
        logger.info("  never converged (%d): %s", len(never_found), ", ".join(never_found))

    return {
        "mode": "CBCT",
        "device": device,
        # Recorded so the artefact carries what makes it reproducible: a
        # markups file plus this line is enough to ask for the same answer
        # again, without knowing how the request was made.
        "seed": seed,
        "prediction_ID": prediction_ID,
        # What drove the selection, and only that: reporting the regions on a
        # run that named landmarks would show a selection the caller never
        # made (they are left at their all-on default in that case).
        "regions": (
            [] if landmarks
            else [catalog.REGION_DISPLAY_NAMES.get(code, code) for code in regions]
        ),
        "landmarks_selected": list(landmarks),
        "landmarks_requested": list(runnable),
        # Named after AMASSS's `structures_without_model` and read by the
        # Slicer module: a landmark listed here means "use another bundle",
        # whereas one in a scan's `landmarks_failed` means "this scan is hard".
        # Kept apart because the fix differs.
        "landmarks_without_model": without_model,
        "landmarks_ungrouped": ungrouped,
        "cases": scan_reports,
        "summary": {
            "total": len(scan_reports),
            "processed": len(processed),
            "failed": len(scan_reports) - len(processed),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }


def _predict_every_scan(scans, scan_reports, weights, runnable, device, budget,
                        preprocessed_dir, output_dir, prediction_ID, seed,
                        pool) -> None:
    """Walk the cohort, recording one report per scan.

    Split out of `predict_landmarks` so the process pool can be opened and
    closed around the whole loop in one `try/finally`, rather than per scan --
    a worker's CUDA context costs about three seconds to build, and a cohort
    would pay it once per patient.
    """
    for scan_index, (scan_path, key) in enumerate(scans, start=1):
        record = {
            "input": os.path.basename(scan_path),
            "status": "pending",
            "landmarks_found": [],
            "landmarks_failed": {},
            "produced": [],
        }
        scan_reports[key] = record
        scan_started = time.monotonic()
        # Position in the batch, never the scan's name: a file name is patient
        # metadata and this server does not write it to a log -- and the same
        # rule is why the progress event carries the counter and nothing else.
        logger.info("scan %d/%d: preprocessing", scan_index, len(scans))
        progress.report(scan_index, len(scans), "scan")

        try:
            broken = _predict_one_scan(
                scan_path=scan_path,
                key=key,
                record=record,
                weights=weights,
                runnable=runnable,
                device=device,
                budget=budget,
                preprocessed_dir=preprocessed_dir,
                output_dir=output_dir,
                prediction_ID=prediction_ID,
                scan_index=scan_index,
                scan_total=len(scans),
                seed=seed,
                pool=pool,
            )
            if broken:
                # A broken executor never recovers: every later `submit`
                # raises. Dropping it here costs the rest of the cohort the
                # width, and saves it the scan-long fallback this one just
                # paid.
                pool = None
            # NOT unconditionally "ok". A per-landmark failure is recorded in
            # `landmarks_failed` and does not raise -- deliberately, since a
            # truncated field of view legitimately misses points and one hard
            # landmark must not cost the other 57. But a scan on which EVERY
            # agent failed placed nothing at all, and calling that a success is
            # how a run that produced no coordinates reports 200.
            if record["landmarks_found"]:
                record["status"] = "ok"
            else:
                record["status"] = "failed"
                record["error"] = (
                    "no landmark could be placed on this scan: "
                    f"{len(record['landmarks_failed'])} of "
                    f"{len(runnable)} agent(s) failed to converge."
                )
        except Exception as exc:
            # One unreadable or hopeless scan must not cost the other 199.
            logger.exception("ALI CBCT failed on one scan")
            record["status"] = "failed"
            record["error"] = str(exc)
        record["duration_seconds"] = round(time.monotonic() - scan_started, 2)
        logger.info(
            "scan %d/%d: %s -- %d found, %d failed, %.0fs",
            scan_index,
            len(scans),
            record["status"],
            len(record["landmarks_found"]),
            len(record["landmarks_failed"]),
            record["duration_seconds"],
        )



def _predict_one_scan(scan_path, key, record, weights, runnable, device, budget,
                      preprocessed_dir, output_dir, prediction_ID,
                      scan_index: int = 1, scan_total: int = 1,
                      seed: int = 0, pool=None) -> bool:
    """Preprocess one scan, run every requested landmark on it, write its file.

    Returns whether the agent pool broke under it, so the cohort stops handing
    work to an executor that can no longer take any.

    Logs progress as it goes. The search is the long part -- 119 landmarks is
    minutes of it -- and without a line in between, a run is indistinguishable
    from a hang. Everything logged here is a COUNT or an anatomical label:
    never a file name, which is patient metadata.
    """
    from .environment import Environment

    scan_work_dir = os.path.join(preprocessed_dir, key.replace(os.sep, "_"))
    started_at = time.monotonic()
    images = _prepare_scan(scan_path, scan_work_dir)
    logger.info(
        "scan %d/%d: preprocessed in %.0fs, searching %d landmark(s)",
        scan_index, scan_total, time.monotonic() - started_at, len(runnable),
    )

    # About ten progress lines per scan, whatever the number of landmarks:
    # one per landmark would be 119 lines here and 23 800 on a 200-scan batch.
    progress_every = max(1, len(runnable) // 10)

    # Loaded in the parent even when the pool does the walking: it is what
    # turns a voxel position back into the scanner's millimetres, and doing
    # that conversion in ONE place is what makes every width agree exactly.
    environment = Environment(
        patient_id=key, padding=_agent_padding(), device=device,
    )

    positions = {}
    try:
        environment.load_images(images)

        search_started = time.monotonic()

        # **The agents walk side by side, and that cannot move a coordinate.**
        # Two properties of this engine make it safe, and both were written
        # for other reasons:
        #
        #   * an agent's random stream is derived from the LANDMARK's name
        #     (`agent.rng_for`), never from a stream shared by the scan, "so
        #     that a landmark's result depends only on (scan, weights, seed)
        #     -- never on which OTHER landmarks were asked for". Order cannot
        #     reach the answer.
        #   * the `Environment` is read-only during a search. Everything the
        #     walk mutates -- position, scale, speed, attempts, the short
        #     memory -- lives on the Agent, and each one has its own Brain.
        #     `predicted_landmarks` is the only dictionary written, one key per
        #     agent, and nothing in this engine ever reads it.
        #
        # A worker is a PROCESS, so it holds its own copy of both. The Brain is
        # built and released around each landmark, so at most one set of
        # networks per worker is resident whatever the number of agents.

        def announce(index):
            """One line every tenth of the batch, wherever the workers are."""
            if index % progress_every and index != len(runnable):
                return
            elapsed = time.monotonic() - search_started
            logger.info(
                "scan %d/%d: %d/%d landmarks, %.0fs elapsed, ~%.0fs left",
                scan_index, scan_total, index, len(runnable), elapsed,
                (elapsed / index) * (len(runnable) - index),
            )

        results, broken = ({}, False)
        if pool is not None and len(runnable) > 1:
            results, broken = _walk_in_pool(pool, images, key, runnable, announce)
            if broken:
                logger.warning(
                    "The agent pool stopped answering after %d of %d "
                    "landmark(s); finishing this scan one at a time. A worker "
                    "was killed -- the host out of memory, most often.",
                    len(results), len(runnable),
                )

        # Whatever the pool did not return, including everything when there is
        # no pool at all. One loop, so the serial path is not a second
        # implementation that could drift from the parallel one.
        for label in runnable:
            if label in results:
                continue
            results[label] = _walk_one(
                label, environment, weights, device, budget, seed
            )
            announce(len(results))

        # Read back in `runnable` order, so `landmarks_found` lists the points
        # in the order the run asked for them and not the order the machine
        # happened to return them.
        for label in runnable:
            voxel_position, error = results[label]
            if error is not None:
                record["landmarks_failed"][label] = error
                continue
            positions[label] = environment.physical_position(
                catalog.SCALE_KEYS[-1], voxel_position
            )
            record["landmarks_found"].append(label)
    finally:
        environment.release()
        # Deleted per scan rather than left to the end of the run: three
        # volumes per scan is gigabytes across a cohort, and holding all of
        # them until `run()` returns is how the output volume fills up.
        shutil.rmtree(scan_work_dir, ignore_errors=True)

    if not positions:
        raise RuntimeError("no landmark converged on this scan")

    # ONE file per scan, holding every region, in the input's own tree.
    destination = os.path.join(
        output_dir,
        os.path.dirname(key),
        f"{scan_stem(os.path.basename(scan_path))}_lm_{prediction_ID}{MARKUPS_EXTENSION}",
    )
    record["produced"].append(write_markups(positions, destination))
    return broken

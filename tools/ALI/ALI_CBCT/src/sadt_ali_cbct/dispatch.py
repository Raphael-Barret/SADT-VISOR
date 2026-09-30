"""Everything ALI_CBCT does before inference: discovery, DICOM conversion, the
run report. `engine.py` only has to know how to place landmarks.

ALI used to be one tool choosing an engine from the data, because a folder can
hold either kind and a DICOM series has no extension to go on. Splitting it in
two moved that question out of the run and into the request: this tool is the
CBCT engine, and an input holding surfaces is refused by name rather than
half-processed. What the split cost is a caller who no longer has "send it and
let ALI work it out"; what it bought is that the two engines no longer share a
virtualenv, and so no longer share a torch version.
"""

import json
import logging
import os
import shutil
import time

from sadt_ali_common.discovery import (
    CBCT,
    VOLUME_EXTENSIONS,
    SURFACE_EXTENSIONS,
    WORK_DIRNAME,
    classify,
    keyed,
    scan_key,
)

from . import catalog as cbct_catalog
from .errors import ToolInputError

logger = logging.getLogger(__name__)


REPORT_NAME = "run_report.json"


class Input:
    """What the input turned out to hold, and how to run it.

    `scans` is a list of `(absolute path, key)` pairs. The key is the path
    relative to the input root and is what identifies a scan everywhere
    afterwards -- in the report and in the output tree. Keying by BASE NAME,
    as the original did, meant two patients called `scan.nii.gz` in different
    subfolders silently overwrote each other twice over: once in the working
    dictionary, once in the flat output folder.
    """

    def __init__(self, mode: str, scans: list, converted_dicom: int = 0):
        self.mode = mode
        self.scans = scans
        self.converted_dicom = converted_dicom



# ---------------------------------------------------------------------------
# Discovery and mode detection
# ---------------------------------------------------------------------------

def _dicom_directories(root: str) -> list:
    """Directories under `root` holding a DICOM series.

    Only directories with no volume or surface file of their own are probed:
    a folder that already contains NIfTI is a folder of scans, not a series,
    and asking GDCM about every directory of a large cohort is slow.
    """
    from . import preprocess

    found = []
    for directory, _subdirs, files in os.walk(root):
        if any(name.lower().endswith(VOLUME_EXTENSIONS + SURFACE_EXTENSIONS) for name in files):
            continue
        if not files:
            continue
        if preprocess.is_dicom_series(directory):
            found.append(directory)
    return sorted(found)


def detect(input_path: str, work_dir: str) -> Input:
    """List the CBCT scans to process, and refuse anything that is not one.

    Before the split this decided WHICH engine ran. It no longer does: this
    tool is the CBCT engine, so the question is only whether the caller sent
    CBCT. Surfaces are named rather than ignored -- silently processing the
    volumes of a mixed folder and dropping the meshes is the failure that
    looks like success.

    No archive is unpacked here. The server extracts a `.zip` before `run()`
    is called -- with the bomb cap and the single-root strip this function used
    to apply itself -- so what arrives is always a real file or directory.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    root = input_path

    if os.path.isfile(root):
        lower = root.lower()
        if lower.endswith(VOLUME_EXTENSIONS):
            return Input(CBCT, [(root, os.path.basename(root))])
        if lower.endswith(SURFACE_EXTENSIONS):
            raise ToolInputError(
                f"'{os.path.basename(root)}' is an intraoral surface. This tool places "
                f"landmarks on CBCT volumes; run ALI_IOS on surfaces."
            )
        raise ToolInputError(
            f"'{os.path.basename(root)}' is not a CBCT volume "
            f"({', '.join(VOLUME_EXTENSIONS)})."
        )

    volumes, surfaces = classify(root)
    dicom_dirs = _dicom_directories(root) if not surfaces else []

    if surfaces and not (volumes or dicom_dirs):
        raise ToolInputError(
            f"This input holds {len(surfaces)} intraoral surface(s) and no CBCT scan. "
            f"Run ALI_IOS on surfaces."
        )
    if surfaces:
        raise ToolInputError(
            f"This input mixes {len(volumes) + len(dicom_dirs)} CBCT scan(s) and "
            f"{len(surfaces)} intraoral surface(s). Send them as two batches, to "
            f"ALI_CBCT and ALI_IOS respectively."
        )

    if volumes or dicom_dirs:
        scans = keyed(volumes, root)
        converted = _convert_dicom(dicom_dirs, root, work_dir)
        scans.extend(converted)
        return Input(CBCT, sorted(scans, key=lambda item: item[1]), converted_dicom=len(converted))

    raise ToolInputError(
        f"No CBCT scan ({', '.join(VOLUME_EXTENSIONS)}) or DICOM series found in the input."
    )



def _convert_dicom(directories: list, root: str, work_dir: str) -> list:
    """Convert each DICOM series to NIfTI; return (path, key) pairs.

    Written into the working directory, never into the input. The original
    created `<input>/NIFTI/` inside the folder the user had selected -- so it
    modified their data, and a second run re-discovered its own output as
    input scans.
    """
    if not directories:
        return []

    from . import preprocess

    destination_root = os.path.join(work_dir, "dicom_converted")
    converted = []
    for directory in directories:
        key = scan_key(directory, root)
        destination = os.path.join(destination_root, f"{key.replace(os.sep, '_')}.nii.gz")
        try:
            preprocess.convert_dicom_series(directory, destination)
        except Exception:
            # A folder GDCM claimed as a series but could not read. Skipped
            # with a log line naming nothing: the folder name is the patient.
            logger.exception("Could not convert a DICOM series")
            continue
        converted.append((destination, f"{key}.nii.gz"))
    logger.info("ALI: converted %d DICOM series", len(converted))
    return converted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def identify(
    input_path: str,
    model_path: str,
    output_dir: str,
    regions=None,
    landmarks=None,
    prediction_ID: str = "Pred",
    device: str = "cuda",
    search_steps: int = 0,
    seed: int = 0,
    num_workers: int = 0,
    sup=None,
) -> dict:
    """Place landmarks on whatever this input holds; return the run report.

    Everything is written under `output_dir`: one markups file per scan, in the
    input's own tree, plus `run_report.json`. Intermediates go in
    `<output_dir>/.ali_work/` and are removed before returning.
    """
    started_at = time.monotonic()

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    try:
        # Walking a cohort and converting DICOM are minutes of work on a large
        # batch, and used to happen in complete silence -- so a run looked hung
        # before it had even started. Counts only, never a file name.
        logger.info("ALI_CBCT: inspecting the input")
        detected = detect(input_path, work_dir)
        logger.info(
            "ALI_CBCT: %s input, %d scan(s)%s",
            detected.mode,
            len(detected.scans),
            f", {detected.converted_dicom} converted from DICOM"
            if detected.converted_dicom
            else "",
        )

        # Imported here, not at module level: the engine pulls torch, monai and
        # itk, and CI imports this package on every PR to publish the schema.
        # That must not cost a CUDA stack.
        from . import engine as cbct_engine

        # An explicit landmark list replaces the regions rather than narrowing
        # them -- see engine.requested_landmarks for why, and for the eight-fold
        # cost that motivates it.
        chosen_landmarks = cbct_catalog.landmark_names(landmarks)
        region_codes = cbct_catalog.region_codes(regions)
        if not chosen_landmarks and not region_codes:
            # The cross-argument rule the schema cannot express.
            raise ToolInputError(
                f"Select at least one region under 'regions' "
                f"({', '.join(cbct_catalog.REGION_NAMES)}), or name the points you "
                f"want under 'landmarks'."
            )

        report = cbct_engine.predict_landmarks(
            scans=detected.scans,
            model_path=model_path,
            regions=region_codes,
            landmarks=chosen_landmarks,
            prediction_ID=prediction_ID,
            output_dir=output_dir,
            work_dir=work_dir,
            device=device,
            search_steps=search_steps,
            seed=seed,
            num_workers=num_workers,
            sup=sup,
        )
        report["dicom_series_converted"] = detected.converted_dicom
    finally:
        # The intermediates are large -- converted DICOM, and every scan
        # preprocessed at two spacings. Removed whether or not the run
        # succeeded, and never from inside the output tree the caller keeps.
        shutil.rmtree(work_dir, ignore_errors=True)

    report["tool"] = "ALI_CBCT"
    # So the report says which weights ran even when nobody read the argument.
    report["model_bundle"] = os.path.basename(str(model_path).rstrip(os.sep))
    report["output_dir"] = output_dir
    report["duration_seconds"] = round(time.monotonic() - started_at, 2)

    # Named `run_report.json` because that is what the Slicer module reads to
    # tell "the model bundle has no such landmark" from "the agent did not
    # converge on this scan" -- two failures that look identical in the scene
    # and need opposite fixes.
    with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "ALI_CBCT finished: %d/%d scan(s) in %.1fs",
        report["summary"]["processed"],
        report["summary"]["total"],
        report["duration_seconds"],
    )
    return report

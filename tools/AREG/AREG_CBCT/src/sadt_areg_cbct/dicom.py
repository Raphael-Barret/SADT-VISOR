"""DICOM -> NIfTI conversion for CBCT input.

Ported from `AREG_CBCT/AREG_CBCT_utils/utils.py::convertdicom2nifti`, with the
three things that made it unusable on a server removed -- the same three ASO's
port removed, since the two CLIs carried the same function:

* it wrote into `<input>/NIFTI/`, i.e. into the caller's own data, which a
  later run then re-discovered as input scans. Everything goes to a scratch
  directory here;
* it only looked one level down (`os.listdir` of the input folder), so a nested
  export was invisible;
* its fallback ran `dicom2nifti.convert_directory` and then took
  `search(output_folder, "nii.gz")[0]` and RENAMED it -- with more than one
  patient already converted that renames an arbitrary earlier file onto the
  current patient's name. The fallback writes into its own empty directory, so
  there is nothing else to pick up.

AREG's own copy rather than an import of `tools/ASO/src/cbct/dicom.py`:
importing another tool's module at load time makes one tool's missing
dependency take both out of the registry. A shared `file_utils` home for it is
the right answer once a third tool needs it.
"""

import logging
import os
import shutil
import tempfile

import SimpleITK as sitk

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Reading this DICOM series needs the dicom2nifti package. Install it with "
    "`pip install -r requirements.txt` (see server/README.md)."
)


def holds_a_series(root: str) -> bool:
    """True when anything under `root` is a readable DICOM series.

    Asked rather than declared. DICOM slices routinely carry no extension at
    all, so a clinician could not tell from a file name either -- which is why
    the panel used to put the question to them, and why the answer being wrong
    was a run that failed for a reason nobody could see. ASO's CBCT engine
    detects it this way and ALI's always has; this is the same question asked
    in the same words.

    AREG's own copy rather than an import of ASO's, for the reason this
    module's docstring already gives about `convert_tree`: importing another
    tool's module at load time makes one tool's missing dependency take both
    out of the registry.

    Stops at the first series found: a cohort of forty patients does not need
    forty answers to a yes/no question.
    """
    reader = sitk.ImageSeriesReader()
    for directory, _subdirs, _names in os.walk(root):
        try:
            if reader.GetGDCMSeriesFileNames(directory):
                return True
        except RuntimeError:
            # GDCM raises on a directory it cannot even scan. That is "no
            # series here", not a failure of the run.
            continue
    return False


def convert_tree(input_root: str, output_root: str) -> str:
    """Convert every DICOM series under `input_root` into `output_root`.

    A directory holding a series becomes `<output_root>/<its relative
    path>.nii.gz`, so a nested export keeps its structure and two patients with
    the same folder name under different parents cannot overwrite each other.

    Returns `output_root`. Raises RuntimeError when the tree holds no series.
    """
    os.makedirs(output_root, exist_ok=True)
    converted = 0

    for directory, _, _file_names in os.walk(input_root):
        series = _series_in(directory)
        if not series:
            continue
        relative = os.path.relpath(directory, input_root)
        name = os.path.basename(directory) if relative != "." else "scan"
        destination = os.path.join(
            output_root, os.path.dirname(relative) if relative != "." else "", f"{name}.nii.gz"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _convert_series(directory, series, destination)
        converted += 1

    if converted == 0:
        raise RuntimeError(
            "No DICOM series found in this input. Send a zip of one folder per patient, "
            "or turn 'dicom_input' off if the scans are already NIfTI/NRRD/GIPL."
        )
    logger.info("AREG: converted %d DICOM series", converted)
    return output_root


def _series_in(directory: str) -> tuple:
    reader = sitk.ImageSeriesReader()
    try:
        return tuple(reader.GetGDCMSeriesFileNames(directory))
    except RuntimeError:
        return ()


def _convert_series(directory: str, series: tuple, destination: str) -> None:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series)
    # A CBCT export routinely carries tags ITK complains about without the read
    # actually failing; the warnings are noise in the server log and could carry
    # patient identifiers.
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    try:
        image = reader.Execute()
    except RuntimeError:
        _convert_with_dicom2nifti(directory, destination)
        return
    finally:
        sitk.ProcessObject_SetGlobalWarningDisplay(True)
    sitk.WriteImage(image, destination, useCompression=True)


def _convert_with_dicom2nifti(directory: str, destination: str) -> None:
    """Fallback for series SimpleITK's GDCM reader refuses."""
    try:
        import dicom2nifti
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise RuntimeError(f"{_INSTALL_HINT} (missing: dicom2nifti)") from exc

    staging = tempfile.mkdtemp(prefix="dicom2nifti_", dir=os.path.dirname(destination))
    try:
        dicom2nifti.convert_directory(directory, staging, compression=True)
        produced = sorted(name for name in os.listdir(staging) if name.lower().endswith(".nii.gz"))
        if not produced:
            raise RuntimeError(
                f"Could not read the DICOM series in '{os.path.basename(directory)}'."
            )
        shutil.move(os.path.join(staging, produced[0]), destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

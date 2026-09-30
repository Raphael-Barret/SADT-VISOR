"""Everything AREG_IOSCBCT does around the registration itself.

Three modes, and they differ only in where the landmarks come from:

    Registration       you supply both sets. Nothing is predicted, no other
                       tool is called, and this needs no GPU at all.
    Semi-Automated     the intraoral meshes are labelled and the landmarks
                       predicted; the CBCT is taken as it is.
    Fully-Automated    the CBCT is oriented first, then both sides predicted.

That progression is why this tool has no engine of its own: each step it does
not do itself is a `sup.run()` into a tool that has one.
"""

import json
import logging
import os
import shutil
import time

import numpy as np

from sadt_areg_common import catalogs
from sadt_areg_common.errors import ToolInputError

from . import geometry, pipeline, progress, tools

logger = logging.getLogger(__name__)

MODALITY = catalogs.MODALITY_IOSCBCT
REPORT_NAME = "AREG_report.json"
WORK_DIRNAME = ".areg_work"


def _read_mesh(path: str):
    """The intraoral mesh with its point data, as pyvista reads it.

    pyvista rather than raw vtk, for two things the registration needs and a
    bare `vtkPolyDataReader` does not hand over as usefully: the `Universal_ID`
    tooth labels, which say which points are crowns, and `compute_normals`,
    which the point-to-plane step measures along. It also transforms the mesh
    with its arrays attached, so the registered file keeps the labels the
    unregistered one carried.
    """
    import pyvista as pv

    return pv.read(path)


def _write_mesh(mesh, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Binary, not ASCII: it round-trips float32 exactly, while ASCII prints six
    # significant digits, and it parses far faster.
    mesh.save(path, binary=True)


def _cbct_surface(scan_path: str, landmark_sets: dict):
    """The CBCT contoured as a surface, plus its verdict on the landmarks.

    Once per patient, not once per arch: contouring a 2.8 million point surface
    is the expensive half of the run, and both arches query the same one. The
    level is read off all of that patient's CBCT landmarks together -- it
    describes the scan, not one jaw, and twelve points make a steadier median
    than six.

    Returns `(None, {}, reason)` when the volume cannot be read. The caller then
    registers on the landmarks alone and the report says the ICP did not run,
    which is worse than refining but better than stopping -- and it is written
    down either way rather than looking like a run that simply had no ICP.
    """
    from . import volume

    merged = {}
    for points in landmark_sets.values():
        merged.update(points)
    try:
        surface, on_enamel = volume.read(scan_path, merged)
    except Exception as exc:  # noqa: BLE001 - a landmark-only run is still a run
        # The class name, not `str(exc)`: readers carry the server's own paths in
        # their messages, and the report goes back to the client. The full
        # traceback stays in the log, where paths belong.
        logger.exception("Could not contour the CBCT of %s", scan_path)
        return None, {}, f"the CBCT could not be contoured ({type(exc).__name__})"
    return surface, on_enamel, None


def _landmarks_by_jaw(directory: str) -> dict:
    """`{relative path: {label: position}}` for every landmark file in a folder.

    Keyed by the path relative to the folder, not by the base name: ALI writes
    one file per scan and MIRRORS the input's tree, so two sites' `P1_U_lm_
    Pred.mrk.json` are two files. Keyed by base name the second silently
    replaced the first, which then made `len(candidates) == 1` -- the "one file
    covers every jaw" fallback -- fire on a folder that held two.
    """
    found = {}
    if not directory or not os.path.isdir(directory):
        return found
    for root, directories, files in os.walk(directory):
        directories.sort()
        for name in sorted(files):
            if not name.lower().endswith(pipeline.LANDMARK_EXTENSIONS):
                continue
            path = os.path.join(root, name)
            points = pipeline.read_landmarks(path)
            if points:
                found[os.path.relpath(path, directory)] = points
    return found


def _for_patient(candidates: dict, patient: str, sole_patient: bool) -> dict:
    """The landmark files that belong to one patient.

    Matching on the jaw token ALONE is what made a two-patient batch register
    the second patient's mesh against the FIRST patient's landmarks: every
    ALI_IOS file carries a `_U` or `_L` token, `sorted()` puts `P1...` before
    `P2...`, and the first jaw match won. Nothing about the result looks wrong
    -- a rigid transform is produced, the mesh is written, the report says
    "ok" -- so the whole batch after patient one is quietly registered onto the
    wrong anatomy.

    A batch holding ONE patient keeps the looser rule. There the two sides
    cannot be confused, and a landmark file named by a convention
    `patient_key` cannot read -- one with no digits in it at all, which is what
    the published reference files look like -- would otherwise stop matching
    anything at all.
    """
    own = {
        name: points
        for name, points in candidates.items()
        if pipeline.patient_key(os.path.basename(name)) == patient
    }
    if own:
        return own
    return candidates if sole_patient else {}


_JAW_TOKENS = {"u", "upper", "l", "lower"}


def _match_landmarks(mesh_path: str, candidates: dict) -> dict:
    """The landmark set belonging to this mesh.

    Two shapes, because the two sides genuinely differ and only one of them can
    name a jaw:

    - **per-jaw files**, which is what ALI_IOS writes (`..._U_lm_Pred.mrk.json`)
      and what upstream's own RegTestFiles carry on both sides. Matched on the
      jaw token, never on sort order: pairing by position is how an upper mesh
      gets registered against a lower arch's points.
    - **one file for everything**, which is what ALI_CBCT writes. A CBCT covers
      both arches in one volume, so its landmark file has no jaw to name. The
      labels themselves carry it -- `UR1O` against `LR1O` -- and
      `shared_landmarks` intersects, so the upper mesh takes the upper points
      out of the same file the lower mesh takes the lower ones from.

    So a jaw match wins when there is one, and a single unlabelled file is
    accepted as covering every jaw rather than refused.
    """
    from sadt_areg_common import pairing

    mesh_tokens = set(pairing.tokens(os.path.basename(mesh_path)))
    for name, points in sorted(candidates.items()):
        if mesh_tokens & set(pairing.tokens(os.path.basename(name))) & _JAW_TOKENS:
            return points

    unlabelled = [
        points for name, points in sorted(candidates.items())
        if not (set(pairing.tokens(os.path.basename(name))) & _JAW_TOKENS)
    ]
    if len(unlabelled) == 1:
        return unlabelled[0]
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    return {}


# The DATA folder this engine's bundles live in, and the orientation reference
# inside it. Written rather than derived, as AREG_CBCT writes its own: which
# folder serves which engine is a deployment fact (the three AREG engines share
# `DATA/AREG/`), and the bundle name is the one the manifest unpacks that
# archive to.
#
# Frankfurt rather than Occlusal, and this is a CHOICE a deployment can
# reverse by naming the other in `cbct_reference`: the two planes carry
# DISJOINT landmark sets, Frankfurt is the frame the shipped test data is
# oriented into, and it is the anatomical convention a CBCT is read in. What it
# must not be is a question asked of a clinician who has no way to know which
# one the rest of the chain expects.
_DATA_NAME = "AREG"
_ORIENTATION_REFERENCE = "CBCT_Gold_Frankfurt_Horizontal_Midsagittal_Plane"


def _own_reference(data_root):
    """The orientation reference this deployment publishes, or ""."""
    if not data_root:
        return ""
    candidate = os.path.join(str(data_root), _DATA_NAME, "models", _ORIENTATION_REFERENCE)
    return candidate if os.path.isdir(candidate) else ""


def register(ios_dir: str, cbct_dir: str, ios_landmark_dir: str, cbct_landmark_dir: str,
             output_dir: str, suffix: str, report: dict, max_dist: float,
             progress_start: float = 0.0) -> None:
    """The registration proper, once every landmark exists.

    `progress_start` is where this phase begins on the run's progress bar. It
    is 0 in the Registration mode, which predicts nothing, and follows the
    waypoints in `tools.py` in the two modes that do -- otherwise a run that
    called no other tool would report itself as more than half done before it
    had registered anything.
    """
    paired, unpaired = pipeline.discover(ios_dir, cbct_dir)
    report["unpaired"] = unpaired

    ios_landmarks = _landmarks_by_jaw(ios_landmark_dir)
    cbct_landmarks = _landmarks_by_jaw(cbct_landmark_dir)
    if not ios_landmarks or not cbct_landmarks:
        raise ToolInputError(
            "Both landmark folders must hold at least one file: found "
            f"{len(ios_landmarks)} intraoral and {len(cbct_landmarks)} CBCT."
        )

    sole_patient = len(paired) == 1
    for index, (patient, data) in enumerate(paired.items(), start=1):
        # Per patient, not per mesh: the inner loop is one or two arches, and
        # the counter is what a watcher can act on. The patient key is built
        # from the caller's file names and never travels in a message.
        progress.report(index, len(paired), "patient", start=progress_start)
        entry = {"cbct": os.path.basename(data["cbct"]), "meshes": {}}
        # Narrowed to this patient BEFORE the jaw is looked at: see _for_patient.
        own_ios = _for_patient(ios_landmarks, patient, sole_patient)
        own_cbct = _for_patient(cbct_landmarks, patient, sole_patient)
        cbct_surface, on_enamel, surface_error = _cbct_surface(data["cbct"], own_cbct)
        if surface_error:
            entry["cbct_surface_error"] = surface_error
        for mesh_path in data["ios"]:
            name = os.path.basename(mesh_path)
            try:
                moving = _match_landmarks(mesh_path, own_ios)
                fixed = _match_landmarks(mesh_path, own_cbct)
                if not moving or not fixed:
                    raise ToolInputError(
                        f"No landmark file matches patient '{patient}' and this mesh's "
                        f"jaw on {'the intraoral' if not moving else 'the CBCT'} side."
                    )
                mesh = _read_mesh(mesh_path)
                matrix, detail = pipeline.register_one(
                    mesh, moving, fixed, cbct_surface, on_enamel, max_dist=max_dist
                )
                destination = os.path.join(
                    output_dir, patient, f"{os.path.splitext(name)[0]}_{suffix}.vtk"
                )
                _write_mesh(mesh.transform(matrix, inplace=False), destination)
                # splitext, not `destination.replace(".vtk", ...)`: str.replace
                # rewrites EVERY occurrence, so a mesh whose own stem carries
                # `.vtk` produced a mangled matrix name beside a correct mesh.
                np.save(os.path.splitext(destination)[0] + "_matrix.npy", matrix)
                entry["meshes"][name] = dict(
                    detail, status="ok", output=os.path.relpath(destination, output_dir)
                )
            except Exception as exc:  # noqa: BLE001 - one mesh must not cost the batch
                logger.exception("AREG_IOSCBCT failed on one mesh")
                entry["meshes"][name] = {"status": "failed", "error": str(exc)}
        registered = [m for m in entry["meshes"].values() if m["status"] == "ok"]
        entry["status"] = "ok" if registered else "failed"
        report["patients"][patient] = entry

    produced = [p for p in report["patients"].values() if p["status"] == "ok"]
    if not produced:
        raise RuntimeError(
            "AREG_IOSCBCT registered no mesh for any patient. The per-mesh errors "
            "are in the report."
        )


def derive_automation(automation: str, ios_landmarks, cbct_landmarks,
                     orient_cbct_first: bool = True) -> tuple:
    """The mode this request really is. Returns `(mode, source)`.

    `source` is "from the data" or "requested". See the note at the call site
    for why only two of the three modes can be read off a folder.
    """
    if automation and automation != catalogs.AUTOMATION_AUTO:
        return automation, "requested"
    if ios_landmarks and cbct_landmarks:
        return catalogs.AUTOMATION_REGISTRATION, "from the data"
    return (catalogs.AUTOMATION_FULLY if orient_cbct_first
            else catalogs.AUTOMATION_SEMI), "from the data"


def main(ios, cbct, output_dir, automation=None, ios_landmarks=None, cbct_landmarks=None,
         cbct_reference=None, landmark_model=None, ios_landmark_model=None,
         crown_model=None, max_dist=None, output_suffix="Reg",
         orient_cbct_first=True, sup=None, data_root=None):
    """Validate, fetch whatever the mode does not supply, then register."""
    started_at = time.monotonic()
    # Two of the three modes are written in the request; the third is a choice.
    #
    # `Registration` is: both landmark sets supplied means predict nothing, there
    # being nothing left to predict. What separates `Fully-Automated` from
    # `Semi-Automated` is NOT in the data -- it is "orient the CBCT first", which
    # nothing in a folder can answer -- so that one is asked as
    # `orient_cbct_first`, a box that says what it does, instead of being hidden
    # inside a three-valued mode nobody could map onto their own files.
    #
    # `cbct_reference` cannot stand in for it either: this deployment resolves
    # its own (`_own_reference`), so one is always present.
    automation, automation_source = derive_automation(
        automation, ios_landmarks, cbct_landmarks, orient_cbct_first
    )
    allowed = catalogs.AUTOMATION_BY_MODALITY[MODALITY]
    if automation not in allowed:
        raise ToolInputError(
            f"'{automation}' is not a mode {MODALITY} has. It offers: {', '.join(allowed)}."
        )

    suffix = (output_suffix or "Reg").strip() or "Reg"
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ToolInputError("'output_suffix' is a name fragment, not a path.")

    output_dir = os.path.abspath(str(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    report = {
        "modality": MODALITY,
        "automation": automation,
        # Which of the two it was: a mode name in a report does not say whether
        # anybody chose it.
        "automation_source": automation_source,
        "output_suffix": suffix,
        "patients": {},
    }

    try:
        ios_root, cbct_root = str(ios), str(cbct)
        ios_lm = str(ios_landmarks) if ios_landmarks else None
        cbct_lm = str(cbct_landmarks) if cbct_landmarks else None

        if automation != catalogs.AUTOMATION_REGISTRATION:
            # Everything the caller did not supply is fetched from the tool that
            # produces it. Checked up front so a request that cannot work comes
            # back in a second rather than after the first prediction.
            for name in ("Crown_Seg", "ALI_IOS", "ALI_CBCT"):
                tools.require(sup, name, f"{automation} IOSCBCT registration")
            if automation == catalogs.AUTOMATION_FULLY:
                tools.require(sup, "ASO", "Fully-Automated IOSCBCT registration")
                # Resolved here rather than asked for: there is one frame the
                # rest of this chain expects, and picking the other silently
                # changes what the landmarks mean.
                cbct_reference = cbct_reference or _own_reference(data_root)
                if not cbct_reference:
                    raise ToolInputError(
                        "Fully-Automated orients the CBCT first, which needs an "
                        f"orientation reference. This deployment publishes none: "
                        f"no '{_ORIENTATION_REFERENCE}' under "
                        f"DATA/{_DATA_NAME}/models/. Add it "
                        "(scripts/setup-models.sh --tool AREG), or name another "
                        "in 'cbct_reference'."
                    )
                cbct_root = tools.orient_cbct(sup, cbct_root, cbct_reference, landmark_model)

            labelled = tools.label_crowns(sup, ios_root, crown_model or "")
            if not ios_lm:
                ios_lm = tools.predict_ios_landmarks(sup, labelled, ios_landmark_model or "")
            if not cbct_lm:
                cbct_lm = tools.predict_cbct_landmarks(sup, cbct_root, landmark_model or "")
            ios_root = labelled

        if not ios_lm or not cbct_lm:
            raise ToolInputError(
                "The Registration mode takes the landmarks already computed: send "
                "both 'ios_landmarks' and 'cbct_landmarks', or use a mode that "
                "predicts them."
            )

        register(
            ios_dir=ios_root, cbct_dir=cbct_root,
            ios_landmark_dir=ios_lm, cbct_landmark_dir=cbct_lm,
            output_dir=output_dir, suffix=suffix, report=report,
            max_dist=float(max_dist) if max_dist else geometry.ICP_MAX_DIST_MM,
            # The last waypoint `tools.py` writes is 0.5; the registration has
            # the rest. Registration mode wrote none of them and starts at 0.
            progress_start=0.0 if automation == catalogs.AUTOMATION_REGISTRATION else 0.6,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    report["duration_seconds"] = round(time.monotonic() - started_at, 2)
    with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("AREG_IOSCBCT: %d patient(s) in %.1fs",
                len(report["patients"]), report["duration_seconds"])
    return output_dir

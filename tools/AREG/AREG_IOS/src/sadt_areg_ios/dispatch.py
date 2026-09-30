"""AREG -- Automated REGistration of two timepoints.

Ported from the Slicer extension's `AREG/` module and its CLI modules
(`AREG_CBCT`, `AREG_IOS`). One tool, two engines, five modes:

|          | Semi-Automated                | Fully-Automated              | Oriented + Fully-Automated |
|----------|-------------------------------|------------------------------|----------------------------|
| **CBCT** | your T1 masks, masked Elastix | AMASSS segments the T1 masks | ASO orients the T1 first   |
| **IOS**  | your segmented meshes         | CrownSeg labels + ASO orients| --                         |

The Slicer envelope is gone entirely: no `<filter-progress>` prints, no
`time.sleep(0.2)` progress theatre, no `sys.exit`, no log file the client
polls, and nothing written into the caller's input tree.

Two entry points, for the same reason AMASSS and ASO have two:

* `register(...)` -> `RegistrationRun`, the real API: the output directory plus
  a structured report. This is what another server-side tool calls.
* `main(...)` -> the output directory's path, the schema adapter `AREG.py` uses.

The Slicer widget built a list of CLI invocations per mode and ran them in
order, passing folders between them. That structure survives, but the steps are
the other packaged tools -- see `tools.py` for how they are called
in-process and through the registry.
"""

import json
import logging
import os
import shutil

from sadt_areg_common.errors import ToolInputError

from sadt_areg_common import catalogs, pairing
from . import progress, tools

logger = logging.getLogger(__name__)

# This tool IS the modality: it is no longer an argument, so the value the
# report carries and the automation table is keyed by is fixed here.
MODALITY = catalogs.MODALITY_IOS

REPORT_NAME = "AREG_report.json"

# Intermediates live here, under the output directory the caller owns, and are
# removed before `register` returns. A surviving `.areg_work/` means a run
# crashed.
WORK_DIRNAME = ".areg_work"


class RegistrationRun:
    """Result of `register()`: where the files are, and what actually happened.

    Reported per patient AND per region, because a CBCT run registering on the
    cranial base and the mandible is two registrations of every patient and one
    of them can fail on its own. The original caught each per-patient exception
    into a log line and finished by printing a count; the archive said nothing.
    """

    def __init__(self, output_dir: str, report: dict):
        self.output_dir = output_dir
        self.report = report

    @property
    def patients(self) -> dict:
        return self.report["patients"]

    @property
    def succeeded(self) -> list:
        return [key for key, entry in self.patients.items() if entry.get("status") == "ok"]



# Which DATA folder this engine's bundles live in, and the two it needs by
# name. Written rather than derived, as the other two AREG engines write their
# own: which folder serves which engine is a deployment fact -- the three AREG
# engines share `DATA/AREG/` -- and the bundle names are the ones the manifest
# unpacks those archives to.
#
# Resolved here rather than asked of the caller. `models/` holds every AREG
# bundle together, around sixty checkpoints, so an unset field arrived as the
# WHOLE folder and the run died on "has to name an entry holding exactly one"
# after it had already segmented and oriented both timepoints. Neither of these
# is a clinical choice: `AREG_model` holds exactly one checkpoint and
# `IOS_Gold_files` is the frame the published test data is oriented into. A
# deployment that wants another still names it, and what it names wins.
_DATA_NAME = "AREG"
_REGISTRATION_BUNDLE = "AREG_model"
_ORIENTATION_REFERENCE = "IOS_Gold_files"


def _own_bundle(data_root, name):
    """The bundle this deployment publishes under that name, or ""."""
    if not data_root:
        return ""
    candidate = os.path.join(str(data_root), _DATA_NAME, "models", name)
    return candidate if os.path.isdir(candidate) else ""


def derive_automation(automation: str, t1_root: str) -> tuple:
    """The mode this request really is, read off the meshes it sends.

    Returns `(mode, source)`, `source` being "from the data" or "requested".

    IOS has two modes and the difference is written in the file: a
    Semi-Automated registration takes crown-segmented meshes, a Fully-Automated
    one labels them with Crown_Seg first. Asking the clinician meant the answer
    could disagree with the folder, and when it did a run either relabelled
    meshes that were already labelled or failed on meshes that were not.

    Read off T1 alone. The two timepoints are one cohort of one patient group
    and are labelled together or not at all; a T2 in a different state is a
    mixed cohort, which `pipeline` already refuses by patient with a message
    about the pair.

    NOT read off `ios_reference`, unlike the CBCT engine's `reference`: this
    deployment resolves its own orientation bundle (`_own_bundle`), so one is
    always present and its presence says nothing about what the caller wants.
    """
    if automation and automation != catalogs.AUTOMATION_AUTO:
        return automation, "requested"

    # Imported here, not at module level: this pulls in vtk, and AREG_IOS is
    # loaded on servers that answer for CBCT alone.
    from . import surfaces

    if surfaces.all_meshes_carry_labels(t1_root):
        return catalogs.AUTOMATION_SEMI, "from the data"
    return catalogs.AUTOMATION_FULLY, "from the data"


def _check_ios(automation, patch, registration_model, reference, mgl_landmarks, height,
               sup=None) -> None:
    if patch not in catalogs.PATCH_CHOICES:
        raise ToolInputError(
            f"Unknown 'ios_patch' {patch!r}. Expected one of: "
            f"{', '.join(catalogs.PATCH_CHOICES)}."
        )

    if patch == catalogs.PATCH_MGL:
        # The mucogingival band is built from landmarks, not predicted by a
        # network, so the palatal checkpoint is not involved at all -- asking
        # for one here is how a user comes to believe this mode needs a model.
        #
        # The landmarks themselves are optional: absent, they are predicted by
        # the landmark tool, which is the whole point of running this on a
        # server. Sending them is for a folder that already has them, which also
        # lets a run be repeated without paying for the prediction again.
        if not mgl_landmarks:
            tools.require(sup, "ALI_IOS", "Registering on the mucogingival line")
        if height is not None and float(height) < 0:
            raise ToolInputError(
                "'mgl_patch_height' is a half-height in millimetres and cannot be "
                "negative. 0 registers on the landmarks alone, without any band."
            )
    elif not registration_model:
        raise ToolInputError(
            "Registering on the palate needs its patch-prediction checkpoint: name "
            "one in 'registration_model' (see GET /tools/AREG_IOS/data)."
        )

    if automation != catalogs.AUTOMATION_FULLY:
        return
    tools.require(sup, "Crown_Seg", "Fully-Automated IOS registration")
    tools.require(sup, "ASO", "Fully-Automated IOS registration")
    if not reference:
        raise ToolInputError(
            "Fully-Automated IOS orients both timepoints before registering, which "
            "needs an orientation reference: name one in 'ios_reference' (see "
            "GET /tools/AREG_IOS/data)."
        )


def _run_ios(
    t1_root, t2_root, automation, registration_model, crown_model, mgl_model, orientation_reference,
    ios_patch, mgl_landmarks_path, mgl_patch_height,
    output_dir, work_dir, suffix, report, sup=None,
) -> None:
    # Imported here rather than at module level: the IOS engine pulls in torch,
    # monai and pytorch3d, and AREG must load (and register CBCT scans) on a
    # server without them.
    from . import landmarks as landmark_files
    from . import butterfly, icp, mgl, net
    from . import pipeline as ios_pipeline
    from . import surfaces

    registered_jaw = catalogs.PATCH_JAW[ios_patch]
    on_palate = ios_patch == catalogs.PATCH_PALATE
    report["patch"] = ios_patch
    report["registered_jaw"] = registered_jaw

    # Only the palatal patch is predicted by a network. The mucogingival band is
    # a spline and a walk over the mesh, so the whole torch/pytorch3d stack is
    # never imported for it -- which is what lets a deployment without pytorch3d
    # register lower arches at full speed while answering 501 for upper ones.
    if on_palate:
        net.check_dependencies()

    prior_transforms: dict = {}
    if automation == catalogs.AUTOMATION_FULLY:
        # Label the crowns, then orient -- the order the Slicer chain used, and
        # the necessary one: ASO's fully-automated IOS mode aligns a mesh by its
        # tooth centroids, so the labels have to exist first.
        t1_root = tools.label_crowns(sup, t1_root, crown_model or "")
        t2_root = tools.label_crowns(sup, t2_root, crown_model or "")
        t1_root = tools.orient_scans(
            sup,
            t1_root, orientation_reference, catalogs.MODALITY_IOS
        )
        t2_root = tools.orient_scans(
            sup,
            t2_root, orientation_reference, catalogs.MODALITY_IOS
        )
        report["labelled_and_oriented"] = True
        prior_transforms = _collect_transforms(t2_root, suffix="Or")

    matched = ios_pipeline.pair(
        t1_root, t2_root, suffix, registered_jaw=registered_jaw, carry_other=on_palate
    )
    report["unmatched"] = matched.report()
    if not matched.matched:
        raise ToolInputError(
            f"No subject has a {registered_jaw.lower()} arch at both timepoints, and "
            f"the {ios_patch} patch lives on that arch. Meshes are paired by name, and "
            f"each one has to say which jaw it is with a token in its name (e.g. "
            f"'P1_T1_{registered_jaw}.vtk' / 'P1_T2_{registered_jaw[0]}.vtk'). "
            f"{len(matched.no_jaw)} mesh(es) named no jaw, "
            f"{len(matched.unpaired)} subject(s) appear at one timepoint only."
        )

    if on_palate:
        predictor = butterfly.PatchPredictor(registration_model)
        painter = ios_pipeline.PalatePainter(predictor)
        report["device"] = predictor.device
        # Which weights placed this patch has to have an answer next to the
        # result: the bundle is a folder and the checkpoint inside it is found,
        # not named.
        report["model_checkpoint"] = os.path.basename(predictor.checkpoint)
    else:
        if mgl_landmarks_path:
            landmark_root = _as_directory(
                mgl_landmarks_path, os.path.join(work_dir, "mgl_landmarks")
            )
            report["mgl_landmarks"] = "sent with the request"
        else:
            # Both timepoints into ONE folder: `landmarks.for_scan` keys on the
            # scan's own name, timepoint included, so T1's and T2's files coexist
            # -- and one call is one model load instead of two.
            landmark_root = os.path.join(work_dir, "mgl_predicted")
            os.makedirs(landmark_root, exist_ok=True)
            for root in (t1_root, t2_root):
                # No model named: ALI picks the hosted bundle matching the input
                # from the models hosted for IT, which is the right default and
                # the only one a caller can express -- AREG's own model list
                # holds the palatal checkpoint and the orientation references,
                # none of which is a landmark bundle.
                _merge_into(
                    tools.predict_mucogingival(sup, root, mgl_model or ""),
                    landmark_root,
                )
            report["mgl_landmarks"] = "predicted by 'ALI_IOS'"

        painter = ios_pipeline.MGLPainter(landmark_root, height=mgl_patch_height)
        report["mgl_patch_height_mm"] = mgl_patch_height

    for index, (key, jaws) in enumerate(sorted(matched.matched.items()), start=1):
        # The counter, never the patient key: the key is built from the file
        # names the caller sent, and a progress message is stored and shown.
        progress.report(index, len(matched.matched), "subject")
        try:
            report["patients"][key] = ios_pipeline.register_patient(
                jaws=jaws,
                painter=painter,
                registered_jaw=registered_jaw,
                output_dir=output_dir,
                relative_key=key,
                suffix=suffix,
                prior_transforms=prior_transforms,
            )
        except (
            icp.RegistrationError,
            surfaces.SurfaceError,
            mgl.PatchError,
            landmark_files.LandmarkError,
        ) as exc:
            report["patients"][key] = {"status": "failed", "reason": str(exc)}


def _collect_transforms(oriented_root: str, suffix: str) -> dict:
    """{patient key: path} of the transforms ASO wrote for the upper arches.

    They are what lets the `.tfm` AREG returns refer to the mesh the CALLER
    sent rather than to the oriented copy AREG made -- see
    `ios.icp.write_transform`.
    """
    from . import surfaces  # for the jaw vocabulary only

    found: dict = {}
    for directory, _, file_names in os.walk(oriented_root):
        relative = os.path.relpath(directory, oriented_root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if not file_name.endswith(".tfm"):
                continue
            if surfaces.jaw_of(file_name) != catalogs.JAW_UPPER:
                continue
            key = os.path.join(
                prefix, pairing.patient_stem(file_name, also_drop=set(catalogs.JAW_TOKENS))
            )
            found.setdefault(key, os.path.join(directory, file_name))
    return found


def _selected(value, choices: dict) -> list:
    """The enabled options of a multichoice argument, in declaration order.

    Accepts the `Selection` validate() produces, a plain dict, or a sequence --
    so `register()` stays directly callable with `["Mandible"]`.
    """
    if value is None:
        return [name for name, on in choices.items() if on]
    if isinstance(value, dict):
        return [name for name in choices if value.get(name)]
    wanted = set(value)
    return [name for name in choices if name in wanted]


def _merge_into(source: str, destination: str) -> None:
    """Copy every file of `source` under `destination`, keeping its tree.

    The two timepoints' landmarks end up in ONE folder on purpose: they are
    matched to their scan by a key that carries the timepoint, so they cannot
    collide, and one folder is one index for the painter to search.
    """
    for directory, _, file_names in os.walk(source):
        relative = os.path.relpath(directory, source)
        target = os.path.join(destination, "" if relative == "." else relative)
        os.makedirs(target, exist_ok=True)
        for file_name in file_names:
            shutil.copy2(os.path.join(directory, file_name), os.path.join(target, file_name))


def _as_directory(path: str, destination: str) -> str:
    """A directory holding the input, whatever shape it arrived in.

    A single uploaded file is linked into a directory of its own rather than
    used from where it landed: main.py streams every upload of a request into
    ONE work directory, so treating a file's parent as an input root would make
    the T2 folder part of the T1 one.
    """
    path = str(path)
    if os.path.isdir(path):
        return path

    os.makedirs(destination, exist_ok=True)
    linked = os.path.join(destination, os.path.basename(path))
    try:
        os.link(path, linked)
    except OSError:
        shutil.copy2(path, linked)
    return destination


def _summarize(report: dict) -> None:
    statuses = [entry.get("status") for entry in report["patients"].values()]
    report["summary"] = {
        "patients": len(statuses),
        "registered": statuses.count("ok"),
        "failed": statuses.count("failed"),
    }
    logger.info(
        "AREG %s %s: %d/%d registered",
        report["modality"],
        report["automation"],
        report["summary"]["registered"],
        report["summary"]["patients"],
    )


def register(
    t1_path: str,
    t2_path: str,
    automation: str,
    orientation_reference: str = None,
    registration_model: str = None,
    crown_model: str = None,
    mgl_model: str = None,
    ios_patch: str = catalogs.PATCH_PALATE,
    mgl_landmarks_path: str = None,
    mgl_patch_height: float = None,
    output_suffix: str = "Reg",
    output_dir: str = None,
    sup=None,
) -> RegistrationRun:
    """Register every T2 under `t2_path` onto its T1 under `t1_path`.

    Each path is a directory or a `.zip`.
    declared in `catalogs.REGION_CHOICES` (CBCT only).
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    t1_root = _as_directory(t1_path, os.path.join(work_dir, "t1_input"))
    t2_root = _as_directory(t2_path, os.path.join(work_dir, "t2_input"))

    # After extraction, because the answer is inside the meshes: a zip has to
    # become a directory before anything can be read out of it.
    automation, automation_source = derive_automation(automation, t1_root)

    report = {
        "modality": MODALITY,
        "automation": automation,
        # Which of the two it was: "Semi-Automated" in a report does not say
        # whether anybody chose it.
        "automation_source": automation_source,
        "output_suffix": output_suffix,
        "patients": {},
    }

    from . import mgl

    _run_ios(
        t1_root=t1_root,
        t2_root=t2_root,
        automation=automation,
        registration_model=registration_model,
        crown_model=crown_model,
        mgl_model=mgl_model,
        orientation_reference=orientation_reference,
        ios_patch=ios_patch,
        mgl_landmarks_path=mgl_landmarks_path,
        mgl_patch_height=(
            mgl.DEFAULT_HEIGHT if mgl_patch_height is None else float(mgl_patch_height)
        ),
        output_dir=output_dir,
        work_dir=work_dir,
        suffix=output_suffix,
        report=report,
        sup=sup,
    )

    # Extracted inputs, converted DICOM, the oriented copies and whatever the
    # tools it drove wrote. Removed whether or not the run succeeded, so what is
    # left under output_dir is results and nothing else.
    shutil.rmtree(work_dir, ignore_errors=True)

    _summarize(report)
    with open(os.path.join(output_dir, REPORT_NAME), "w") as handle:
        json.dump(report, handle, indent=2)
    return RegistrationRun(output_dir, report)


def main(
    automation,
    t1,
    t2,
    ios_reference=None,
    registration_model=None,
    crown_model=None,
    mgl_model=None,
    ios_patch=None,
    mgl_landmarks=None,
    mgl_patch_height=None,
    output_suffix="Reg",
    output_dir=None,
    sup=None,
    data_root=None,
) -> str:
    """Translate the schema's arguments into `register()` and return its output
    directory, which main.py zips and streams.

    Every cross-argument rule is checked HERE, before any file is read: a
    request that cannot work must come back in a second, not after an hour of
    registration. `require` in tools.py is part of that: a mode that needs
    another tool fails at the door when there is no supervisor to reach it.
    """
    automation = str(automation)
    suffix = (output_suffix or "Reg").strip() or "Reg"
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ToolInputError("'output_suffix' is a name fragment, not a path.")

    allowed = catalogs.AUTOMATION_BY_MODALITY.get(MODALITY, ())
    if automation not in allowed:
        raise ToolInputError(
            f"'{automation}' is not a mode {MODALITY} has. {MODALITY} offers: "
            f"{', '.join(allowed)}."
        )

    patch = str(ios_patch or catalogs.PATCH_PALATE)
    # What the caller named wins; what it left empty this deployment fills. The
    # checks below then see a real bundle and their message stays about what is
    # missing from the DEPLOYMENT rather than about a field the panel no longer
    # shows.
    registration_model = registration_model or _own_bundle(
        data_root, _REGISTRATION_BUNDLE)
    reference = ios_reference or _own_bundle(data_root, _ORIENTATION_REFERENCE)
    # The mode is read off the meshes, and the meshes are not extracted yet --
    # so what is checked here is FULLY's requirements, which are the superset:
    # everything Semi needs, plus Crown_Seg, ASO and an orientation reference.
    # Checking the superset keeps the promise this function's docstring makes,
    # that a request which cannot work comes back in a second rather than after
    # an hour of registration.
    #
    # The one case it costs: a deployment WITHOUT Crown_Seg, handed meshes that
    # are already labelled, is refused although Semi-Automated would have run.
    # Naming 'Semi-Automated' in `automation` is the override for exactly that,
    # and `tools.require` says so.
    checked = (catalogs.AUTOMATION_FULLY
               if automation in ("", catalogs.AUTOMATION_AUTO) else automation)
    _check_ios(checked, patch, registration_model, reference,
               mgl_landmarks, mgl_patch_height, sup)

    run = register(
        t1_path=str(t1),
        t2_path=str(t2),
        automation=automation,
        orientation_reference=str(reference) if reference else None,
        registration_model=str(registration_model) if registration_model else None,
        crown_model=str(crown_model) if crown_model else None,
        mgl_model=str(mgl_model) if mgl_model else None,
        ios_patch=patch,
        mgl_landmarks_path=str(mgl_landmarks) if mgl_landmarks else None,
        mgl_patch_height=mgl_patch_height,
        output_suffix=suffix,
        output_dir=output_dir,
        sup=sup,
    )

    return run.output_dir

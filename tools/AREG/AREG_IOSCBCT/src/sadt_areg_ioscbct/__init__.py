"""AREG_IOSCBCT -- register an intraoral scan onto a CBCT of the same patient.

NOT longitudinal, unlike its two siblings: AREG_CBCT and AREG_IOS take a
baseline and a follow-up of one modality, this takes ONE timepoint imaged two
ways. Upstream's own test set says so -- `P001_T2_U.vtk` beside
`P_0001_T2.nii.gz`, both T2 -- which is why the arguments are `ios` and `cbct`
rather than `t1` and `t2`.

**It has no engine of its own, and that is the design.** No torch, no pytorch3d,
no nnUNet: the landmarks come from ALI_CBCT and ALI_IOS, the tooth labels from
Crown_Seg and the orientation from ASO, each in its own virtualenv, reached
through the supervisor. Containing both stacks would mean one environment
holding torch 2.8 for one half and 2.11 for the other -- the exact defect the
AREG and ALI splits removed. What is left here is geometry.
"""

from pathlib import Path
from typing import Literal

from sadt_areg_common import pairing

from . import pipeline
from .dispatch import main


# What each path argument can read, for the client's file dialog and for the
# server's refusal of an upload that contradicts it. DERIVED from the tables
# the pipeline already registers against -- a second spelling of them would be
# a second thing to keep in step.
#
# Without this the intraoral picker offered a folder of CBCT volumes, and the
# mistake surfaced as "No patient has both an intraoral scan and a CBCT" a
# minute into the run rather than at the click.
ACCEPTS = {
    "ios": pipeline.SURFACE_EXTENSIONS,
    "cbct": pairing.SCAN_EXTENSIONS,
    "ios_landmarks": pipeline.LANDMARK_EXTENSIONS,
    "cbct_landmarks": pipeline.LANDMARK_EXTENSIONS,
}


def run(
    ios: Path,
    cbct: Path,
    output_dir: Path,
    # "From the data" is the default and names no mode: `dispatch.derive_automation`
    # reads it off the request -- both landmark sets supplied is Registration,
    # which predicts nothing. Whether to orient the CBCT first is `orient_cbct_first`
    # below, because no folder can answer it. Naming a mode is an override.
    automation: Literal[
        "From the data", "Registration", "Semi-Automated", "Fully-Automated"
    ] = "From the data",
    ios_landmarks: Path = "",
    cbct_landmarks: Path = "",
    cbct_reference: Path = "",
    landmark_model: Path = "",
    ios_landmark_model: Path = "",
    crown_model: Path = "",
    max_dist: float = 0.0,
    # The one thing about this mode that a folder cannot answer, asked as what it
    # is instead of hidden inside a three-valued `automation`: orienting the CBCT
    # into a standard frame before the cross-modality registration is a clinical
    # decision, not a property of the files. On by default -- it is what
    # "Fully-Automated" did, and the frame the rest of the chain expects.
    #
    # Read only when no landmarks are supplied: with both sets in hand there is
    # nothing to predict and nothing to orient for.
    orient_cbct_first: bool = True,
    output_suffix: str = "Reg",
    *,
    sup=None,
    data_root=None,
) -> Path:
    """Register an intraoral scan onto a CBCT of the same patient.

    Args:
        ios: The intraoral surfaces (.vtk/.stl), one folder, searched
            recursively. Upper and lower arches are registered separately and
            matched to their landmarks by the jaw token in the name.
        cbct: The CBCT volumes, one folder. Paired to the intraoral scans on the
            digits of the patient identifier, the two modalities being named by
            different conventions.
        output_dir: Where the registered meshes, their 4x4 matrices and
            `AREG_report.json` are written. Nothing is written outside it.
        automation: Registration takes both landmark sets and predicts nothing --
            no other tool is called and no GPU is needed. Semi-Automated labels
            the crowns and predicts both sets. Fully-Automated orients the CBCT
            first, which needs a reference.
        ios_landmarks: Registration mode. Your own intraoral landmarks.
        cbct_landmarks: Registration mode. Your own CBCT landmarks.
        cbct_reference: The frame the CBCT volumes are oriented onto before
            registering.
        landmark_model: The CBCT landmark bundle, for the modes that predict.
        ios_landmark_model: The intraoral landmark bundle.
        crown_model: The crown-labelling checkpoint, for the modes that label.
        max_dist: How far a point may be from its nearest neighbour and still
            count as an ICP correspondence, in millimetres. 0 uses 1.5, which is
            upstream's.
        orient_cbct_first: Put the CBCT in the Frankfurt horizontal and
            mid-sagittal frame before the cross-modality registration. One frame
            only, unlike the CBCT-to-CBCT engine which offers two: the landmarks
            the rest of this chain reads are defined in this one, so another
            would silently change what they mean. Read only when no landmarks
            are supplied, there being nothing to orient for once they are.
        output_suffix: Added to each output name, e.g. `scan_Reg.vtk`.

    Returns:
        The output directory.
    """
    output_dir = Path(output_dir)
    main(
        ios=ios, cbct=cbct, output_dir=output_dir, automation=automation,
        ios_landmarks=ios_landmarks, cbct_landmarks=cbct_landmarks,
        cbct_reference=cbct_reference, landmark_model=landmark_model,
        ios_landmark_model=ios_landmark_model, crown_model=crown_model,
        max_dist=max_dist, output_suffix=output_suffix,
        orient_cbct_first=orient_cbct_first, sup=sup,
        data_root=data_root,
    )
    return output_dir

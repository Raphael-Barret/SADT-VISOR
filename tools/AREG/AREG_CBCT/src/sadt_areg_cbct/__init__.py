"""AREG_CBCT -- register a follow-up CBCT onto its baseline.

elastix, rigid, restricted to the anatomy that has not changed between the two
timepoints: the cranial base, the mandible or the maxilla, taken as masks.

Split out of the former single `AREG`, which served both modalities from one
schema and one virtualenv. The reason is the same as ALI's: the intraoral
engine needs pytorch3d and therefore torch 2.11, this one needs neither, and
while they shared an environment neither could be pinned without the other.
They now share only `sadt_areg_common` -- the patient-key convention, the
catalogs, and the scan-extension table -- which has no dependencies at all.

The masks this registers on, and the orientation both timepoints must share,
come from other tools reached through the supervisor; see the CBCT half of
`tools.py`. `AREG_CBCT -> ASO -> ALI_CBCT` is the deepest chain in the family.
"""

from pathlib import Path
from typing import Literal

from .dispatch import main


# `t1`, `t2` and `t1_masks` deliberately declare NO extensions, and it is not
# an omission. The AREG facade composes this engine with the intraoral one, and
# `t1` there is a SURFACE: declaring volumes here made the two disagree about
# what one name means, and the facade -- rightly -- refused to publish at all
# ("'AREG' cannot publish 't1'"). Saying what this engine reads needs an
# `accepts` the facade can vary per MODE, the same capability its file pickers
# need to be scoped per mode. Until then the engine's own panel filters
# nothing, and `pairing.is_scan_file` is what actually decides.

def run(
    t1: Path,
    t2: Path,
    output_dir: Path,
    # "From the data" is the default and names no mode: `dispatch.derive_automation`
    # reads it off what the request supplies -- masks are Semi-Automated, none is
    # Fully-Automated, and the frame named in `orientation` below chooses the
    # oriented variant.
    # Naming one of the three is an override and still obeyed. Spelled out because
    # `Literal` takes literals only; a test asserts it against `catalogs`.
    automation: Literal[
        "From the data",
        "Semi-Automated", "Fully-Automated", "Oriented + Fully-Automated",
    ] = "From the data",
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from catalogs.REGION_CHOICES. That makes this a second declaration of the
    # same set, which is the thing this contract otherwise avoids, so a test
    # asserts the two agree.
    regions: list[
        Literal["Cranial base", "Mandible", "Maxilla"]
    ] = ["Cranial base"],
    t1_masks: Path = "",
    # The second group of boxes the original module shows, and it is NOT the
    # regions above: those decide what the registration is masked to, these
    # decide what comes back to look at. Spelled out for the same reason
    # `regions` is -- `Literal` cannot be built from the catalog.
    segmentations: list[
        Literal[
            "Cranial base", "Cervical vertebra", "Mandible", "Maxilla",
            "Skin", "Upper airway",
        ]
    ] = [],
    segmentation_model: Path = "",
    segmentation_label: int = 0,
    reference: Path = "",
    landmark_model: Path = "",
    dicom_input: bool = False,
    # WHICH FRAME the scans come back in, and the one thing about the automated
    # path that no folder can answer. Naming a frame is what asks for the T1 to
    # be oriented first; "Leave as scanned" is the default and asks for nothing.
    #
    # Two frames rather than a yes/no, because the two published bundles carry
    # disjoint landmark sets and mean different things -- Frankfurt horizontal
    # for the cranial base, the occlusal plane for the occlusion. A boolean here
    # could only have said "the default one", which is the defect this replaced.
    #
    # Read only when no `t1_masks` are sent: with masks in hand there is nothing
    # to segment and nothing to orient for.
    orientation: Literal[
        "Leave as scanned", "Frankfurt horizontal", "Occlusal plane"
    ] = "Leave as scanned",
    output_suffix: str = "Reg",
    *,
    sup=None,
    data_root=None,
) -> Path:
    """Register a follow-up CBCT onto its baseline, so the two can be compared.

    Args:
        t1: The baseline scans -- one volume or a folder of them, searched
            recursively. A DICOM series is converted when `dicom_input` is set.
        t2: The follow-up scans, paired to T1 by patient key.
        output_dir: Where the registered scans, their transforms and
            `AREG_report.json` are written. Nothing is written outside it.
        automation: Left on its default the mode is read off the request --
            masks mean Semi-Automated, none means Fully-Automated, and
            the frame named in `orientation` picks the oriented variant.
            Naming a mode
            overrides that: Semi-Automated takes your own masks, Fully-Automated
            segments them, Oriented + Fully-Automated orients both timepoints
            first, which needs a reference.
        regions: The anatomy to register on -- what has NOT changed between the
            timepoints. The one argument a clinician must actually think about.
        t1_masks: Your own T1 segmentation masks, instead of having them
            segmented for you.
        segmentations: Anatomy to segment and return beside the registration,
            for the modes that segment. Independent of `regions`: ticking the
            skin does not register on it, and registering on the mandible does
            not return a mandible you can open. None by default -- a
            registration run returns a registration.
        segmentation_model: The mask model bundle, for the modes that segment.
            Left empty -- which is what a panel sends -- the AMASSS bundle this
            deployment publishes for AREG is used, there being no second answer
            to the question.
        segmentation_label: Which label value in the masks to register on.
        reference: The frame the scans are oriented onto before registering.
        landmark_model: The landmark bundle that orientation step predicts
            with.
        dicom_input: The inputs are DICOM series rather than volumes. Detected
            from the data when left off; this forces the conversion.
        orientation: The anatomical frame the scans are returned in. "Leave as
            scanned" keeps the scanner's own frame and skips the orientation
            step; naming a frame orients the T1 onto that reference first, which
            costs a landmark prediction per patient. The two are not
            interchangeable: Frankfurt horizontal is defined by
            Ba, S, N, RPo, LPo, ROr and LOr, the occlusal plane by
            ANS, IF, PNS, UL6O, UR1O and UR6O, so the measurement made
            afterwards does not say the same thing. Read only when no masks are
            supplied.
        output_suffix: Added to each output name, e.g. `scan_Reg.nii.gz`.

    Returns:
        The output directory.
    """
    # itk-elastix and SimpleITK are imported inside the engine: CI imports this
    # module on every PR to publish the schema, and that must not cost them.
    return main(
        t1=t1,
        t2=t2,
        output_dir=output_dir,
        automation=automation,
        cbct_regions=regions,
        t1_masks=t1_masks,
        segmentations=segmentations,
        segmentation_model=segmentation_model,
        segmentation_label=segmentation_label,
        cbct_reference=reference,
        landmark_model=landmark_model,
        dicom_input=dicom_input,
        orientation=orientation,
        output_suffix=output_suffix,
        sup=sup,
        data_root=data_root,
    )

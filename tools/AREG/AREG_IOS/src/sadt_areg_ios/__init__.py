"""AREG_IOS -- register a follow-up intraoral scan onto its baseline.

A patch of the arch that does not move with growth or treatment -- the palate,
or the band around the mucogingival line -- matched by ICP.

Split out of the former single `AREG`. The reason is concrete: this engine
needs pytorch3d, which ships as a wheel built against one exact torch
(`+pt2110cu128`), so it is pinned to torch 2.11. The CBCT engine needs neither
and has no reason to move, and while the two shared a virtualenv neither could
be pinned without the other. They now share only `sadt_areg_common`, which has
no dependencies at all.

The tooth labels and the mucogingival landmarks this needs come from other
tools reached through the supervisor; see `tools.py`.
"""

from pathlib import Path
from typing import Literal

from .dispatch import main


def run(
    t1: Path,
    t2: Path,
    output_dir: Path,
    # "From the data" is the default and names no mode: `dispatch.derive_automation`
    # reads it off the meshes -- ones that already carry their tooth labels are
    # Semi-Automated, ones that do not are Fully-Automated. Naming either is an
    # override and still obeyed.
    automation: Literal[
        "From the data", "Semi-Automated", "Fully-Automated"
    ] = "From the data",
    ios_reference: Path = "",
    patch: Literal[
        "Palate (upper arch)", "Mucogingival line (lower arch)"
    ] = "Palate (upper arch)",
    registration_model: Path = "",
    crown_model: Path = "",
    mgl_model: Path = "",
    mgl_landmarks: Path = "",
    mgl_patch_height: float = 0.0,
    output_suffix: str = "Reg",
    *,
    sup=None,
    data_root=None,
) -> Path:
    """Register a follow-up intraoral scan onto its baseline, so the two compare.

    Args:
        t1: The baseline meshes -- one surface or a folder of them, searched
            recursively. Each must say its jaw in its name.
        t2: The follow-up meshes, paired to T1 by patient key.
        output_dir: Where the registered meshes, their transforms and
            `AREG_report.json` are written. Nothing is written outside it.
        automation: Semi-Automated takes meshes that already carry their tooth
            labels and orientation; Fully-Automated labels and orients them
            first, through the crown-segmentation and orientation tools.
        ios_reference: The frame the arches are oriented onto before registering.
        patch: Which part of the arch to match on -- the palate for an upper
            arch, the band around the mucogingival line for a lower one.
        registration_model: The model that finds the palatal patch. Not used by
            the mucogingival patch, which is built from landmarks and involves
            no network at all.
        crown_model: The checkpoint the crown-labelling tool runs with. A mesh
            that already carries its tooth-label array needs none.
        mgl_model: The landmark bundle the mucogingival line is predicted
            from, when no landmarks are sent.
        mgl_landmarks: Your own 13 landmarks per lower scan, marking the
            mucogingival line, instead of having them predicted.
        mgl_patch_height: How far the band extends from the mucogingival
            line, in millimetres. 0 uses the engine's default.
        output_suffix: Added to each output name, e.g. `scan_Reg.vtk`.

    Returns:
        The output directory.
    """
    # torch, pytorch3d and monai are imported inside the engine: CI imports this
    # module on every PR to publish the schema, and that must not cost a CUDA
    # stack.
    return main(
        t1=t1,
        t2=t2,
        output_dir=output_dir,
        automation=automation,
        ios_reference=ios_reference,
        ios_patch=patch,
        registration_model=registration_model,
        crown_model=crown_model,
        mgl_model=mgl_model,
        mgl_landmarks=mgl_landmarks,
        mgl_patch_height=mgl_patch_height,
        output_suffix=output_suffix,
        sup=sup,
        data_root=data_root,
    )

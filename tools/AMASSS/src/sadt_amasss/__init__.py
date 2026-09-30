"""AMASSS -- Automatic Multi-Anatomical Skull Structure Segmentation.

One nnUNet v2 model per anatomical structure, run over one scan or a whole
folder of them. The pipeline is in pipeline.py; only `run` is public.
"""

from pathlib import Path
from typing import Literal

from .catalog import merge_modes, structure_codes
from .pipeline import segment


# Two caveats that used to sit in `run()`'s docstring, and so in the panel:
# the display names the old schema used ("Cranial base", ...) are still accepted
# alongside the codes, so a client that has not moved keeps working; and a
# single-structure run always writes the separate form, a "merged" volume of one
# structure being just that structure. Both are contracts, neither is something
# to read while choosing what to segment.
def run(
    scans: Path,
    model: Path,
    output_dir: Path,
    # The options are spelled out because `Literal` takes literals only -- it
    # cannot be built from catalog.STRUCTURE_CODES. That makes this a second
    # declaration of the same set, which is the thing this contract otherwise
    # avoids, so a test asserts the two agree.
    structures: list[
        Literal["MAND", "MAX", "CB", "CV", "UAW", "SKIN", "CBMASK", "MANDMASK", "MAXMASK"]
    ] = ["MAND", "MAX", "CB", "CV", "UAW"],
    merge: list[Literal["MERGED", "SEPARATE"]] = ["MERGED"],
    prediction_ID: str = "Pred",
    generate_surface: bool = False,
    surface_smoothing: int = 5,
    surface_decimation: int = 90,
    device: Literal["cuda", "cpu"] = "cuda",
    tile_step_size: float = 0.5,
    gpu_resampling: bool = True,
    num_workers: int = 0,
    *,
    sup=None,
) -> Path:
    """Segment craniofacial structures on a CBCT scan.

    Args:
        scans: One oriented CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/
            .gipl.gz), or a folder of them for a batch. Folders are searched
            recursively, and files that look like a previous AMASSS output are
            skipped so a folder can be re-run in place.
        model: The model bundle: one subfolder per structure code (MAND/, MAX/,
            ...), each holding an nnUNet v2 model.
        output_dir: Where results are written -- one `<scan>_<ID>_SegOut/`
            folder per scan, plus `AMASSS_report.json`. Nothing is written
            outside it.
        structures: What to segment. One with no model in the bundle is
            reported rather than failing the run.
        merge: Which form the output takes; both may be given.
        prediction_ID: Suffix used in output names, e.g. `scan_Pred_MAND.nii.gz`.
        generate_surface: Also export a 3D surface (.vtk) beside each
            segmentation.
        surface_smoothing: Smoothing iterations for the surfaces (0-95).
            Ignored without generate_surface.
        surface_decimation: Percentage of surface triangles to drop (0-99).
            Marching cubes runs on the original scan grid, so a 0.33 mm CBCT
            gives a triangle per voxel face -- 3.5 M across a five-structure
            run -- for detail the mask does not have, being accurate to about
            half a voxel. 90 drops nine triangles in ten and moves the
            cranial-base surface by 0.059 mm on average (max 0.692 mm); 0 keeps
            the raw mesh. Ignored without generate_surface.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.
        tile_step_size: nnUNet's sliding-window overlap; the window advances by
            patch_size times this. It DOES move the segmentation (0.7 measures
            Dice 0.995 against 0.5), so it is left at nnUNet's own default.
        gpu_resampling: Resample on the GPU instead of nnUNet's scipy splines.
            Roughly seven times less time in resampling, which is where a run
            actually goes. Ignored on CPU and for a bundle whose plans pin a
            non-default resampler. Set false for bit-identical nnUNet output.
        num_workers: How many structures to predict at once. 0 lets the server
            decide from the room it reserved for this run, which is the normal
            case; a number is a ceiling on that, never a floor over it. One
            structure is one nnUNet model resident on the card, and a run is
            two thirds preprocessing on a single core -- which is the idle time
            this fills.

    Returns:
        The output directory, holding one folder per scan plus the run report.
    """
    # torch, nnunetv2, SimpleITK and vtk are imported inside the pipeline: CI
    # imports this module on every PR to publish the schema, and that must not
    # cost a CUDA stack.
    output_dir = Path(output_dir)
    segment(
        input_path=Path(scans),
        model_path=Path(model),
        output_dir=output_dir,
        structures=structure_codes(structures),
        merge=merge_modes(merge),
        prediction_ID=prediction_ID,
        generate_surface=generate_surface,
        surface_smoothing=surface_smoothing,
        surface_decimation=surface_decimation,
        device=device,
        tile_step_size=tile_step_size,
        gpu_resampling=gpu_resampling,
        num_workers=num_workers,
        sup=sup,
    )
    return output_dir

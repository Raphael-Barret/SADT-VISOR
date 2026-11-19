"""CrownSeg -- per-tooth labelling of intraoral surface scans.

The pipeline is in pipeline.py; only `run` is public.
"""

import sys
from pathlib import Path
from typing import Literal

from .errors import ToolInputError
from .pipeline import array_name_for, segment_crowns

# The DATA folder this tool's weights live in: its own name. Written rather
# than derived, because which folder holds which bundle is a deployment fact
# and a wrong guess is a directory that is simply not there.
_DATA_NAME = "Crown_Seg"


def _own_models(data_root):
    """`<root>/Crown_Seg/models`, or a refusal a caller can act on.

    `find_checkpoint` takes it from here: it already walks a folder and refuses
    to choose between two vintages, so this only has to say WHICH folder.
    """
    if data_root is None:
        raise ToolInputError(
            "No 'model' given and no data root to look in. Name the checkpoint, "
            "or run this through a server that publishes one."
        )
    return Path(data_root) / _DATA_NAME / "models"


# "mps" is published on macOS only: every other platform keeps the options it always had.
_DEVICES = Literal["cuda", "cpu", "mps"] if sys.platform == "darwin" else Literal["cuda", "cpu"]
# What a caller that names no device gets. Not only the published default: a
# tool another tool runs through `sup.run` is sent none, and `cuda` on a Mac is
# a request the CPU answers, many times slower than the Apple GPU it could use.
_DEFAULT_DEVICE = "mps" if sys.platform == "darwin" else "cuda"


def run(
    meshes: Path,
    output_dir: Path,
    # After `output_dir` and optional, which is the shape that lets a neighbour
    # ask for labelled crowns WITHOUT naming weights. Left required, the four
    # supervised calls that omit it -- AREG IOS and IOSCBCT pass `crown_model
    # or ""`, and that argument is optional on their own schemas -- died on
    # `TypeError: run() missing 1 required positional argument: 'model'`.
    # Empty means "my own", resolved below from this tool's data folder.
    model: Path = "",
    # Empty means "named after the numbering", which is the only default that
    # cannot lie about what the array holds. See pipeline._ARRAY_NAMES.
    array_name: str = "",
    suffix: str = "Seg",
    numbering: Literal["Universal", "FDI"] = "Universal",
    skip_segmented: bool = True,
    device: _DEVICES = _DEFAULT_DEVICE,
    num_workers: int = 2,
    *,
    data_root=None,
) -> Path:
    """Label every tooth of an intraoral scan with its dental number.

    Args:
        meshes: One intraoral surface (.vtk/.stl), or a folder of them for a
            batch. Folders are searched recursively and the output mirrors the
            input tree, so two patients whose meshes share a file name stay
            apart.
        model: The crown-segmentation checkpoint (a .pth file), or a folder
            holding it -- the published one is named for the training run that
            produced it, `<date>_val-loss<number>.pth`, and is found inside a
            folder however deeply it is filed. That is what lets a caller hand
            over a whole models directory without knowing where the file sits;
            a folder holding two of them is refused, naming both. Left empty,
            this tool uses the checkpoint installed for it.
        output_dir: Where the labelled meshes are written, plus
            `run_report.json`. Nothing is written outside it.
        array_name: Name of the point-data array the labels are written to.
            Left empty it follows `numbering` -- `Universal_ID` or `FDI_ID` --
            so the name can never disagree with the integers underneath it.
            Every tool that reads these meshes looks for the Universal spelling,
            which is what makes an FDI mesh refused downstream rather than read
            as a different set of teeth.
        suffix: Added to each output name, e.g. `arch_Seg.vtk`.
        numbering: Universal or FDI tooth numbering. This changes the integers
            written into the array, not the mesh, and whatever consumes the
            result has to agree -- the report records which was used.
        skip_segmented: Pass a mesh that already carries labels through
            unchanged instead of spending minutes re-predicting it. That is
            what makes a mixed batch of raw and pre-segmented meshes one call.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.
        num_workers: Accepted for compatibility and no longer used: it sized the
            data loader of the engine this tool used to call. Meshes are read in
            the calling process.

    Returns:
        The output directory, holding the labelled meshes and the run report.
        The report lists every mesh that now carries labels under
        `segmented_meshes` -- that is what a caller sequencing this before ALI
        reads.
    """
    # shapeaxi, torch and vtk are imported inside the pipeline: CI imports this
    # module on every PR to publish the schema, and the segmentation stack is
    # an optional extra that a CI venv deliberately does not have.
    output_dir = Path(output_dir)
    segment_crowns(
        input_path=str(meshes),
        # Its OWN checkpoint when nobody named one. A caller that names one is
        # pinning which weights ran and is obeyed; a caller that does not --
        # a panel with no model field, a neighbour asking through the
        # supervisor -- gets this tool's, found from this tool's data folder.
        model_path=str(model or _own_models(data_root)),
        output_dir=str(output_dir),
        array_name=array_name or array_name_for(numbering),
        suffix=suffix,
        fdi=numbering == "FDI",
        skip_segmented=skip_segmented,
        device=device,
        num_workers=num_workers,
    )
    return output_dir

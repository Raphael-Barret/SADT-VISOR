"""ALI_CBCT -- Automatic Landmark Identification on CBCT volumes.

One deep-RL agent per landmark walks the volume at 1 mm and then at 0.3 mm
until it converges on the point. Writes one Slicer markups file per scan.

Split out of the former single `ALI` tool, which chose between this engine and
the intraoral one from the data. The two shared nothing but their output
format and their input vocabulary -- both now in `sadt_ali_common` -- while
being forced to share a virtualenv, and therefore a torch version. The
intraoral engine needs torch 2.11 for pytorch3d; this one has no reason to
move. That is what the split buys.
"""

from pathlib import Path
from typing import Literal

from .dispatch import identify
from .errors import ToolInputError


# The DATA folder this tool's weights live in. Its own name, except that
# ALI_CBCT and ALI_IOS are two engines behind the ALI facade and share one.
# Written rather than derived: which is which is a deployment fact, and a wrong
# guess is a folder that is simply not there.
_DATA_NAME = "ALI"


def _own_models(data_root):
    """`<root>/ALI/models`, or a refusal a caller can act on."""
    if data_root is None:
        raise ToolInputError(
            "No 'model' given and no data root to look in. Name the model "
            "bundle, or run this through a server that publishes one."
        )
    return Path(data_root) / _DATA_NAME / "models"

# What this engine appends to the name it was handed, and it is a CONSTANT.
# It used to be an argument, `prediction_ID`, defaulting to "Pred". That made
# the marker a property of the REQUEST rather than of the tool, so nothing
# downstream could know it: pairing a scan with its landmarks, and working out
# which results belong to one patient, both have to strip a marker they can
# predict. A caller wanting to label a run labels the output FOLDER, which
# costs nobody a guess.
#
# Published through `OUTPUT_SUFFIXES` below, which is how the server learns it
# without holding a table of dental names.
PREDICTION_ID = "Pred"

# Only the part that identifies the tool, not the whole written name: a file
# is `<patient>_lm_Pred.mrk.json`, and cutting at `_lm` is what recovers the
# patient whatever follows it.
OUTPUT_SUFFIXES = ("_lm_Pred", "_lm")

# What a reader may do with what this engine produced, and therefore whether a
# stop after a call to it is somewhere to come BACK to. Landmarks are dragged
# and saved back to their own file, which is the whole reason a reader is shown
# them: an orientation computed from a bad point cannot be fixed where it is
# looked at.
REVIEW_KIND = "landmarks"


def run(
    input: Path,
    output_dir: Path,
    # After `output_dir` and optional, which is the shape that lets a neighbour
    # ask for landmarks WITHOUT naming weights. Left required, a supervised call
    # omitting it died on `TypeError: run() missing 1 required positional
    # argument: 'model'` -- the same failure AREG's four supervised calls show.
    # Empty means "my own", resolved below from this tool's data folder.
    model: Path = "",
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from catalog.REGION_NAMES. That makes this a second declaration of the
    # same set, which is the thing this contract otherwise avoids, so a test
    # asserts the two agree.
    regions: list[
        Literal["Cranial base", "Upper", "Lower", "Impacted canine"]
    ] = ["Cranial base", "Upper", "Lower", "Impacted canine"],
    landmarks: list[
        Literal[
            "Ba", "S", "N", "RPo", "LPo", "RFZyg", "LFZyg", "C2", "C3", "C4", "RInfOr",
            "LInfOr", "LMZyg", "RPF", "LPF", "PNS", "ANS", "A", "UR3O", "UR1O", "UL3O",
            "UR6DB", "UR6MB", "UL6MB", "UL6DB", "IF", "ROr", "LOr", "RMZyg", "RNC",
            "LNC", "UR7O", "UR5O", "UR4O", "UR2O", "UL1O", "UL2O", "UL4O", "UL5O",
            "UL7O", "UL7R", "UL5R", "UL4R", "UL2R", "UL1R", "UR2R", "UR4R", "UR5R",
            "UR7R", "UR6MP", "UL6MP", "UL6R", "UR6R", "UR6O", "UL6O", "UL3R", "UR3R",
            "UR1R", "RCo", "RGo", "Me", "Gn", "Pog", "PogL", "B", "LGo", "LCo", "LR1O",
            "LL6MB", "LL6DB", "LR6MB", "LR6DB", "LAF", "LAE", "RAF", "RAE", "LMCo",
            "LLCo", "RMCo", "RLCo", "RMeF", "LMeF", "RSig", "RPRa", "RARa", "LSig",
            "LARa", "LPRa", "LR7R", "LR5R", "LR4R", "LR3R", "LL3R", "LL4R", "LL5R",
            "LL7R", "LL7O", "LL5O", "LL4O", "LL3O", "LL2O", "LL1O", "LR2O", "LR3O",
            "LR4O", "LR5O", "LR7O", "LL6R", "LR6R", "LL6O", "LR6O", "LR1R", "LL1R",
            "LL2R", "LR2R", "UR3OIP", "UL3OIP", "UR3RIP", "UL3RIP",
        ]
    ] = [],
    device: Literal["cuda", "cpu"] = "cuda",
    search_steps: int = 0,
    seed: int = 0,
    num_workers: int = 0,
    *,
    data_root=None,
    # It calls no other tool, and it is no longer unused: `sup.channels()` is
    # how this engine learns how many agents the machine will pay for at once.
    sup=None,
) -> Path:
    """Place anatomical landmarks on a CBCT scan.

    Args:
        input: One CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), or a
            folder of them for a batch. Folders are searched recursively, and a
            DICOM series found inside one is converted automatically. An input
            holding intraoral surfaces is refused by name rather than half
            processed -- run ALI_IOS on those.
        model: The model bundle, holding `<landmark>/<scale>/*.pth` folders.
        output_dir: Where results are written -- one `<scan>_lm_<ID>.mrk.json`
            per scan, mirroring the input's own folder tree, plus
            `run_report.json`. Nothing is written outside it.
        regions: Anatomical regions to predict. Every region is on by default:
            a landmark whose weights the bundle lacks costs a line in the run
            report, whereas a region left off by default is one nobody finds.
        landmarks: Predict exactly these landmarks -- naming any of them
            REPLACES the region selection rather than narrowing it, which is
            what lets a caller ask for the seven points it needs instead of
            running 58 agents to use them. Left empty, `regions` decides, which
            is what a client showing no region control relies on.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.
        search_steps: Forward passes one agent may spend looking for its
            landmark before it is reported as not found. 0 uses the engine's
            default, since there is no nullable type in the schema to express
            "unset" with. The same number on every device: a step is a forward
            pass, so the bound no longer moves with the hardware or the load.
        seed: Seed the agents respawn from, so a run is reproducible. An agent
            that steps out of the volume restarts from a random position, and
            for a landmark at the edge of the field of view that position
            decides the answer -- so on the default the same scan gives the
            same landmarks every time, and changing it is how to see how stable
            a point actually is.
        num_workers: How many landmarks to search at once. 0 lets the server
            decide from the room it reserved for this run, which is the normal
            case; a number is a ceiling on that, never a floor over it. One
            agent holds its own networks on the card, so the budget clamps the
            answer long before the request does. It cannot change a
            coordinate: an agent's random stream is derived from its
            landmark's name, so the answer never depends on which other
            landmarks were asked for, nor in what order they ran.

    Returns:
        The output directory, holding the markups files and the run report.
    """
    # torch, monai, itk and SimpleITK are all imported inside the engine: CI
    # imports this module on every PR to publish the schema, and that must not
    # cost a CUDA stack.
    output_dir = Path(output_dir)
    identify(
        input_path=str(input),
        # Its OWN bundle when nobody named one. A caller that names a bundle is
        # pinning which weights ran and is obeyed; a caller that does not --
        # a neighbour asking through the supervisor for landmarks, say -- gets
        # this tool's, found from this tool's own data folder. Composing that
        # path was the CALLER's job until now, which meant ASO holding a name
        # for ALI's storage and a copy of its weights beside its own.
        model_path=str(model or _own_models(data_root)),
        output_dir=str(output_dir),
        regions=regions,
        landmarks=landmarks,
        prediction_ID=PREDICTION_ID,
        device=device,
        search_steps=search_steps,
        num_workers=num_workers,
        sup=sup,
        seed=seed,
    )
    return output_dir

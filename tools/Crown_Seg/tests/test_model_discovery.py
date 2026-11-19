"""`model` as a DIRECTORY: finding the crown checkpoint inside one.

`model` used to be a single `.pth` and nothing else. It still is when a caller
passes one -- the first test here is that promise -- but the server hands a
hosted-model argument the whole of `DATA/<tool>/models/` when nobody named a
bundle, and ALI_IOS passes on the directory IT was handed when it asks this
tool to label a mesh mid-run. Neither knows where the file sits inside.

The folder those two hand over is shared: `DATA/ALI/models/` holds 243 ALI
checkpoints beside the one crown checkpoint. So the discovery has to RECOGNISE
a crown checkpoint rather than take any `.pth`, and the fixture below is the
real bundle's file names, not invented ones.
"""

import os

import pytest

from sadt_crownseg import pipeline
from sadt_crownseg.errors import ToolInputError

from test_run import stub_engine_run, write_surface  # noqa: F401 - fixture + helper


# The published crown checkpoint, and the neighbours it has to be told apart
# from. Both ALI bundles are named exactly as `scripts/data-manifest.yml` stages
# them and as DATA/ALI/models/ holds them today.
CROWN = pipeline.PUBLISHED_CHECKPOINT
ALI_IOS_WEIGHTS = (
    "ALI_IOS_Models/Upper_O_model.pth",
    "ALI_IOS_Models/Lower_O_model.pth",
    "ALI_IOS_Models/Upper_C_model.pth",
    "ALI_IOS_Models/Lower_C_model.pth",
    "ALI_IOS_Models/Lower_MG_v6.pth",
)
ALI_CBCT_WEIGHTS = (
    "ALI_CBCT_Models/Cranial_Base/Ba/1/Ba_Net_1.pth",
    "ALI_CBCT_Models/Cranial_Base/Ba/0-3/Ba_Net_0-3.pth",
    "ALI_CBCT_Models/Lower_Right_Teeth/LR7R/1/LR7R_Net_1.pth",
)


def write_bundle(root, names):
    """A models folder holding `names`, each a path relative to it."""
    for name in names:
        path = os.path.join(str(root), *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"not a real checkpoint")
    return str(root)


# ---------------------------------------------------------------------------
# The four cases
# ---------------------------------------------------------------------------

def test_a_plain_pth_path_is_used_exactly_as_before(tmp_path):
    """The promise this change must not break.

    Returned unchanged, and without the name being looked at at all: a caller
    who points `model` at a file has already answered the question discovery
    exists to answer, and second-guessing them would refuse every checkpoint
    trained since the published one.
    """
    checkpoint = tmp_path / "some_other_name.pth"
    checkpoint.write_bytes(b"not a real checkpoint")

    assert pipeline.find_checkpoint(str(checkpoint)) == str(checkpoint)


def test_one_crown_checkpoint_in_a_folder_is_the_one_used(tmp_path):
    bundle = write_bundle(tmp_path / "models", [f"Crown_Seg_Model/{CROWN}"])

    found = pipeline.find_checkpoint(bundle)

    assert found == os.path.join(bundle, "Crown_Seg_Model", CROWN)


def test_two_crown_checkpoints_are_named_rather_than_chosen_between(tmp_path):
    """Which model vintage ran must never be a surprise.

    Picking here -- the first, the newest, the largest -- would leave that
    unrecorded, so both are named and the caller says which. The names are
    relative to the folder: this message reaches the client verbatim and the
    server's own paths are not its business.
    """
    bundle = write_bundle(
        tmp_path / "models",
        [f"Crown_Seg_Model/{CROWN}", "old/09-30-21_val-loss0.212.pth"],
    )

    with pytest.raises(ToolInputError) as raised:
        pipeline.find_checkpoint(bundle)

    message = str(raised.value)
    assert CROWN in message
    assert "09-30-21_val-loss0.212.pth" in message
    assert bundle not in message, "the server's own path travelled to the client"
    assert str(tmp_path) not in message


def test_a_folder_with_no_crown_checkpoint_says_what_is_missing(tmp_path):
    bundle = write_bundle(tmp_path / "models", ALI_IOS_WEIGHTS + ALI_CBCT_WEIGHTS)

    with pytest.raises(ToolInputError) as raised:
        pipeline.find_checkpoint(bundle)

    message = str(raised.value)
    assert "setup-models.sh" in message
    # What to look for, and an example of it: the fix has to be actionable
    # without reading this source.
    assert pipeline.CROWN_CHECKPOINT_MARKER in message
    assert CROWN in message


def test_a_path_that_is_neither_file_nor_folder_is_still_not_found(tmp_path):
    """The message every existing caller already gets for a bad `model`."""
    with pytest.raises(ToolInputError, match="checkpoint not found"):
        pipeline.find_checkpoint(str(tmp_path / "absent.pth"))


# ---------------------------------------------------------------------------
# Beside somebody else's weights
# ---------------------------------------------------------------------------

def test_the_ali_bundles_are_not_mistaken_for_crown_weights(tmp_path):
    """The case that decides the rule.

    `DATA/ALI/models/` is what ALI_IOS hands over, and it holds 243 ALI
    checkpoints. Taking "the only .pth" or "the first .pth" would segment a
    patient's crowns with a landmark network.
    """
    bundle = write_bundle(
        tmp_path / "models",
        ALI_IOS_WEIGHTS + ALI_CBCT_WEIGHTS + (f"Crown_Seg_Model/{CROWN}",),
    )

    assert pipeline.find_checkpoint(bundle) == os.path.join(
        bundle, "Crown_Seg_Model", CROWN
    )


@pytest.mark.parametrize(
    "name, recognised",
    [
        ("07-21-22_val-loss0.169.pth", True),
        # `_` and `-` are interchangeable in the wild, so both spellings of the
        # token are read.
        ("2026-01-01_val_loss0.100.pth", True),
        ("Upper_O_model.pth", False),
        ("Lower_MG_v6.pth", False),
        ("Ba_Net_0-3.pth", False),
        ("checkpoint_final.pth", False),
        # Right name, wrong kind of file: shapeaxi is handed a `.pth`.
        ("07-21-22_val-loss0.169.ckpt", False),
    ],
)
def test_what_counts_as_a_crown_checkpoint(name, recognised):
    assert pipeline._is_crown_checkpoint(name) is recognised


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_a_folder_works_as_the_model_argument_of_a_whole_run(tmp_path, stub_engine_run):
    """What ALI_IOS's supervised call actually does."""
    write_surface(tmp_path / "cohort" / "arch.vtk")
    bundle = write_bundle(
        tmp_path / "models", ALI_IOS_WEIGHTS + (f"Crown_Seg_Model/{CROWN}",)
    )

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
    )

    assert report["summary"]["segmented"] == 1
    # shapeaxi was handed the checkpoint, never the folder.
    assert stub_engine_run[0]["model"] == os.path.join(bundle, "Crown_Seg_Model", CROWN)


def test_the_report_says_which_checkpoint_ran(tmp_path, stub_engine_run):
    """`model` no longer answers that on its own: it may be a folder holding
    several vintages, and the one picked out of it is what a result has to be
    attributable to."""
    write_surface(tmp_path / "cohort" / "arch.vtk")
    bundle = write_bundle(tmp_path / "models", [f"Crown_Seg_Model/{CROWN}"])

    report = pipeline.segment_crowns(
        input_path=str(tmp_path / "cohort"),
        model_path=bundle,
        output_dir=str(tmp_path / "out"),
    )

    assert report["checkpoint"] == CROWN

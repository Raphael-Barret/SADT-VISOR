"""Finding the mask a registration is confined to, and applying it.

A mask decides which anatomy the two timepoints are aligned on, so matching one
to the wrong scan is not a crash: it is a confident registration of the wrong
thing. The original matched region tokens as SUBSTRINGS -- `"cb" in
basename.lower()` makes every file whose name contains CBCT a cranial-base
mask, `"max"` matches a patient called MAX_01, and `"md"` matches almost
anything.
"""

import os

import numpy as np
import pytest
import SimpleITK as sitk

from conftest import full_mask, phantom, write
from sadt_areg_cbct import elastix, pipeline
from sadt_areg_common import catalogs, pairing


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_a_mask_has_to_say_both_what_it_is_and_what_it_covers(tmp_path):
    image = phantom(size=16)
    write(image, str(tmp_path / "P1_MAND.nii.gz"))         # no segmentation token
    write(image, str(tmp_path / "P2_seg.nii.gz"))          # no region token
    write(image, str(tmp_path / "P3_MAND_seg.nii.gz"))     # both
    assert list(pairing.discover_masks(str(tmp_path), "MAND")) == ["P3"]


@pytest.mark.parametrize("token", catalogs.MASK_TOKENS)
def test_every_segmentation_token_the_family_writes_is_recognised(tmp_path, token):
    """AMASSS writes `seg`, the Slicer module wrote `mask`, and a caller's own
    export says `pred` or `segmentation`. Recognising three of four refuses a
    batch that is perfectly well named."""
    write(phantom(size=16), str(tmp_path / f"P1_MAND_{token}.nii.gz"))
    assert list(pairing.discover_masks(str(tmp_path), "MAND")) == ["P1"]


@pytest.mark.parametrize("region", sorted(catalogs.REGION_TOKENS))
def test_every_region_can_be_named_by_any_of_its_own_tokens(tmp_path, region):
    for index, token in enumerate(catalogs.REGION_TOKENS[region]):
        write(phantom(size=16), str(tmp_path / f"P{index}_{token}_seg.nii.gz"))
    found = pairing.discover_masks(str(tmp_path), region)
    assert len(found) == len(catalogs.REGION_TOKENS[region])


def test_a_region_token_is_a_whole_token_not_a_substring(tmp_path):
    """`"cb" in "P1_CBCT_seg"` is what made every CBCT a cranial-base mask."""
    write(phantom(size=16), str(tmp_path / "P1_CBCT_seg.nii.gz"))
    assert pairing.discover_masks(str(tmp_path), "CB") == {}


def test_one_regions_mask_is_not_another_regions(tmp_path):
    """Each region is a separate registration, so a mask crossing over means
    two of them silently register on the same anatomy."""
    write(phantom(size=16), str(tmp_path / "P1_MAND_seg.nii.gz"))
    assert list(pairing.discover_masks(str(tmp_path), "MAND")) == ["P1"]
    assert pairing.discover_masks(str(tmp_path), "MAX") == {}
    assert pairing.discover_masks(str(tmp_path), "CB") == {}


def test_a_mask_keys_to_the_patient_its_scan_keys_to(tmp_path):
    """`P1_T1_MAND_seg.nii.gz` has to key to `P1`, the same key
    `P1_T1_scan.nii.gz` gets, or nothing ever matches."""
    write(phantom(size=16), str(tmp_path / "P1_T1_MAND_seg.nii.gz"))
    assert list(pairing.discover_masks(str(tmp_path), "MAND")) == ["P1"]


def test_amasss_own_output_names_are_recognised(tmp_path):
    """What the Fully-Automated path actually has to read back."""
    write(phantom(size=16), str(tmp_path / "P1_T1_scan_seg_MANDMASK.nii.gz"))
    assert list(pairing.discover_masks(str(tmp_path), "MAND")) == ["P1"]


def test_a_hidden_file_is_not_a_mask(tmp_path):
    """A macOS AppleDouble sits beside every file of a copied folder and reads
    as an empty volume."""
    write(phantom(size=16), str(tmp_path / "P1_MAND_seg.nii.gz"))
    with open(str(tmp_path / "._P1_MAND_seg.nii.gz"), "wb") as handle:
        handle.write(b"\x00\x05\x16\x07")
    assert len(pairing.discover_masks(str(tmp_path), "MAND")) == 1


def test_mask_discovery_is_recursive(tmp_path):
    write(phantom(size=16), str(tmp_path / "a" / "b" / "P1_MAND_seg.nii.gz"))
    assert list(pairing.discover_masks(str(tmp_path), "MAND")) == [
        os.path.join("a", "b", "P1")
    ]


# ---------------------------------------------------------------------------
# Choosing between several folders
# ---------------------------------------------------------------------------

def test_an_explicit_mask_folder_beats_one_found_next_to_the_scans(tmp_path):
    """A mask can come from three places, and the caller naming one is the
    strongest statement of intent there is."""
    write(phantom(size=16), str(tmp_path / "explicit" / "P1_MAND_seg.nii.gz"))
    write(phantom(size=16), str(tmp_path / "beside" / "P1_MAND_seg.nii.gz"))

    found = pipeline.find_masks(
        [str(tmp_path / "explicit"), str(tmp_path / "beside")], "MAND"
    )
    assert found["P1"].startswith(str(tmp_path / "explicit"))


def test_a_root_that_does_not_exist_is_skipped_rather_than_fatal(tmp_path):
    """The Semi-Automated path appends the T1 folder unconditionally, and the
    automated one appends whatever AMASSS returned."""
    write(phantom(size=16), str(tmp_path / "masks" / "P1_MAND_seg.nii.gz"))
    found = pipeline.find_masks([None, "/nonexistent", str(tmp_path / "masks")], "MAND")
    assert list(found) == ["P1"]


def test_a_mask_under_amasss_own_output_folder_still_finds_its_scan(tmp_path):
    """AMASSS writes one `<scan>_<id>_SegOut/` directory per scan, so a mask
    discovered under it keys to `P1_seg_SegOut/P1` while its scan keys to
    `P1`."""
    write(
        phantom(size=16),
        str(tmp_path / "P1_T1_scan_seg_SegOut" / "P1_T1_scan_seg_MANDMASK.nii.gz"),
    )
    found = pipeline.find_masks([str(tmp_path)], "MAND", scan_keys=["P1"])
    assert os.path.basename(found["P1"]) == "P1_T1_scan_seg_MANDMASK.nii.gz"


def test_the_leaf_fallback_is_only_taken_for_a_key_that_is_really_missing(tmp_path):
    """It must not reroute a patient whose mask was found where it belongs."""
    write(phantom(size=16), str(tmp_path / "P1_MAND_seg.nii.gz"))
    write(phantom(size=16), str(tmp_path / "elsewhere" / "P1_MAND_seg.nii.gz"))

    found = pipeline.find_masks([str(tmp_path)], "MAND", scan_keys=["P1"])
    assert found["P1"] == str(tmp_path / "P1_MAND_seg.nii.gz")


def test_an_ambiguous_leaf_is_not_guessed(tmp_path):
    """Two subjects genuinely called P1 in different folders must not borrow
    each other's mask."""
    for site in ("siteA", "siteB"):
        write(phantom(size=16), str(tmp_path / site / "P1_T1_MAND_seg.nii.gz"))
    found = pipeline.find_masks([str(tmp_path)], "MAND", scan_keys=["P1"])
    assert "P1" not in found


def test_a_patient_with_no_mask_anywhere_is_simply_absent(tmp_path):
    write(phantom(size=16), str(tmp_path / "P1_MAND_seg.nii.gz"))
    found = pipeline.find_masks([str(tmp_path)], "MAND", scan_keys=["P1", "P2"])
    assert set(found) == {"P1"}


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------

def test_a_mask_of_a_different_spacing_is_refused_with_both_spacings(tmp_path):
    """Same voxel count, different millimetres: `CopyInformation` would make
    the two agree on paper and register several millimetres off."""
    image = phantom(size=24, spacing=0.8)
    mask = full_mask(phantom(size=24, spacing=0.5))
    with pytest.raises(elastix.RegistrationError) as raised:
        elastix.apply_mask(image, mask)
    message = str(raised.value)
    assert "0.5" in message and "0.8" in message


def test_a_float_drift_in_the_origin_is_copied_across_rather_than_refused():
    """A mask that made a round trip through a segmentation tool comes back
    a fraction of a micron off. Identical sampling means that difference is
    drift, not geometry."""
    image = phantom(size=24)
    mask = full_mask(image)
    mask.SetOrigin(tuple(value + 1e-9 for value in image.GetOrigin()))

    masked, note = elastix.apply_mask(image, mask)
    assert note is None
    assert masked.GetOrigin() == pytest.approx(image.GetOrigin())


def test_the_masked_image_keeps_the_scans_own_geometry():
    image = phantom(size=24)
    masked, _note = elastix.apply_mask(image, full_mask(image))
    assert masked.GetSize() == image.GetSize()
    assert masked.GetSpacing() == pytest.approx(image.GetSpacing())
    assert masked.GetOrigin() == pytest.approx(image.GetOrigin())


def test_label_zero_means_the_whole_mask_not_the_background():
    """`segmentation_label` is 0 by default, and 0 is also a label value. It
    has to mean "all of it" -- registering on the background is registering on
    nothing."""
    image = phantom(size=24)
    array = np.zeros(sitk.GetArrayViewFromImage(image).shape, np.uint8)
    array[6:14] = 1
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)

    masked, note = elastix.apply_mask(image, mask, label=0)
    kept = sitk.GetArrayViewFromImage(masked)
    assert note is None                 # two values only: nothing to warn about
    assert kept[6:14].any()
    assert not kept[:6].any() and not kept[14:].any()


def test_a_single_label_mask_says_nothing_and_a_three_label_one_does():
    image = phantom(size=24)
    for labels, expected in (((1,), False), ((1, 2), True), ((1, 2, 3), True)):
        array = np.zeros(sitk.GetArrayViewFromImage(image).shape, np.uint8)
        for index, value in enumerate(labels):
            array[4 + index * 4: 8 + index * 4] = value
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(image)
        _masked, note = elastix.apply_mask(image, mask, label=0)
        assert bool(note) is expected, labels


def test_the_label_refusal_lists_what_the_mask_actually_holds():
    """So the fix is "use 2", not "try again"."""
    image = phantom(size=24)
    array = np.zeros(sitk.GetArrayViewFromImage(image).shape, np.uint8)
    array[6:14] = 2
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)

    with pytest.raises(elastix.RegistrationError) as raised:
        elastix.apply_mask(image, mask, label=7)
    message = str(raised.value)
    assert "no label 7" in message
    assert "0, 2" in message
    assert "segmentation_label" in message


# ---------------------------------------------------------------------------
# DICOM, asked of the data
# ---------------------------------------------------------------------------
# `dicom_input` used to be a check box a clinician had to tick. DICOM slices
# routinely carry no extension, so they could not tell from a file name either,
# and answering wrong produced a run that failed for a reason nobody could see.
# The flag survives as an override; the answer is now read off the folder.

def _write_dicom_series(directory, size=(8, 8, 6)):
    """A minimal but genuinely readable CT series."""
    os.makedirs(str(directory), exist_ok=True)
    array = np.zeros(size[::-1], dtype=np.int16)
    array[2:5, 2:6, 2:6] = 500
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 1.0))

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    series_uid = "1.2.826.0.1.3680043.2.1125.1234567890"
    for index in range(image.GetDepth()):
        slice_image = image[:, :, index]
        position = "\\".join(
            str(value) for value in image.TransformIndexToPhysicalPoint((0, 0, index))
        )
        for tag, value in (
            ("0008|0060", "CT"), ("0020|000e", series_uid), ("0020|0032", position),
            ("0020|0013", str(index)), ("0028|0030", "0.5\\0.5"), ("0018|0050", "1.0"),
            ("0020|0037", "1\\0\\0\\0\\1\\0"),
        ):
            slice_image.SetMetaData(tag, value)
        writer.SetFileName(os.path.join(str(directory), f"slice{index:03d}.dcm"))
        writer.Execute(slice_image)
    return str(directory)


def test_a_folder_of_slices_is_recognised_as_dicom(tmp_path):
    from sadt_areg_cbct import dicom

    _write_dicom_series(tmp_path / "patientA")

    assert dicom.holds_a_series(str(tmp_path))


def test_a_nested_export_is_found_too(tmp_path):
    """One level down was all the original looked, so a site/patient export was
    invisible and the run read no scans at all."""
    from sadt_areg_cbct import dicom

    _write_dicom_series(tmp_path / "siteA" / "patientA" / "scan")

    assert dicom.holds_a_series(str(tmp_path))


def test_a_folder_of_volumes_is_not(tmp_path):
    """The other half of the statement: if this answered yes to everything, the
    two tests above would pass against a detector that always says DICOM -- and
    every ordinary cohort would be sent through the converter, which refuses
    with 'No DICOM series found in this input'."""
    from sadt_areg_cbct import dicom

    write(phantom(size=16), str(tmp_path / "T1" / "P1_T1.nii.gz"))

    assert not dicom.holds_a_series(str(tmp_path))


def test_an_empty_folder_is_not_dicom(tmp_path):
    from sadt_areg_cbct import dicom

    assert not dicom.holds_a_series(str(tmp_path))

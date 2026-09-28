"""Resampling: the interpolation rule, and the geometry that must survive it.

Nearest neighbour for a segmentation, linear otherwise. Interpolating a label
map linearly produces labels that were never in it, and a label is an
anatomical structure -- 2 between 1 and 3 is a different bone, not a rounding
error.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_automatrix import pipeline


# A direction that is not the identity, so a test asserting geometry survives a
# resampling can tell "preserved" from "defaulted": ITK hands out the identity
# for an image that never had a direction set.
ROTATED_DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _shift(amount):
    return sitk.TranslationTransform(3, (amount, amount, amount))


# ---------------------------------------------------------------------------
# The interpolation rule
# ---------------------------------------------------------------------------

def test_nearest_neighbour_keeps_the_label_set_it_was_given(label_image):
    labels = pipeline.resample(label_image(), _shift(0.3), is_segmentation=True)

    assert set(np.unique(sitk.GetArrayFromImage(labels))) <= {0, 1, 3}


def test_linear_interpolation_invents_the_label_between_them(label_image):
    """The other half of the same statement: if this stopped blending, the test
    above would pass against a broken tool."""
    scan = pipeline.resample(label_image(), _shift(0.3), is_segmentation=False)

    assert 2 in set(np.unique(sitk.GetArrayFromImage(scan)))


def test_a_half_voxel_shift_could_not_have_shown_that(label_image):
    """Why the offset is 0.3 and not 0.5, kept as an assertion so nobody
    "tidies" it: without a reference the output origin moves by the translation
    too, so a 0.5 shift samples exactly one voxel away, on the grid, and the
    two interpolators return the very same array."""
    image = label_image()

    as_labels = pipeline.resample(image, _shift(0.5), is_segmentation=True)
    as_scan = pipeline.resample(image, _shift(0.5), is_segmentation=False)

    assert np.array_equal(sitk.GetArrayFromImage(as_labels),
                          sitk.GetArrayFromImage(as_scan))


@pytest.mark.parametrize("offset", [0.1, 0.25, 0.3, 0.75, 1.4])
def test_no_off_grid_shift_makes_nearest_neighbour_invent_a_label(label_image, offset):
    labels = pipeline.resample(label_image(), _shift(offset), is_segmentation=True)

    assert set(np.unique(sitk.GetArrayFromImage(labels))) <= {0, 1, 3}


def test_the_rule_holds_on_a_reference_grid_too(label_image):
    """A reference changes where the samples fall, not how they are taken."""
    reference = sitk.Image(10, 10, 10, sitk.sitkInt16)
    reference.SetSpacing((0.7, 0.7, 0.7))

    labels = pipeline.resample(label_image(), _shift(0.3), reference, True)
    scan = pipeline.resample(label_image(), _shift(0.3), reference, False)

    assert set(np.unique(sitk.GetArrayFromImage(labels))) <= {0, 1, 3}
    assert 2 in set(np.unique(sitk.GetArrayFromImage(scan)))


def test_the_identity_transform_returns_the_image_unchanged(label_image):
    image = label_image()

    resampled = pipeline.resample(image, sitk.Transform(3, sitk.sitkIdentity))

    assert np.array_equal(sitk.GetArrayFromImage(resampled),
                          sitk.GetArrayFromImage(image))


def test_what_falls_outside_the_input_is_zero(label_image):
    """The default pixel value, and it must be 0: a segmentation's background
    is 0 and a scan resampled onto a value of its own would gain a slab of
    fabricated tissue at its edge."""
    far = pipeline.resample(label_image(), _shift(50.0), is_segmentation=True)

    assert set(np.unique(sitk.GetArrayFromImage(far))) == {0}


# ---------------------------------------------------------------------------
# Geometry, with no reference: the input's own grid
# ---------------------------------------------------------------------------

def test_the_grid_is_the_input_s_own(label_image):
    image = label_image(spacing=(0.5, 0.75, 1.25))

    resampled = pipeline.resample(image, _shift(1.0))

    assert resampled.GetSize() == image.GetSize()
    assert resampled.GetSpacing() == (0.5, 0.75, 1.25)


def test_only_the_origin_moves_and_it_moves_by_the_transform(label_image):
    """Exactly: origin (10, 20, 30) plus the translation (1, 2, 3)."""
    image = label_image(origin=(10.0, 20.0, 30.0))

    resampled = pipeline.resample(image, sitk.TranslationTransform(3, (1.0, 2.0, 3.0)))

    assert resampled.GetOrigin() == (11.0, 22.0, 33.0)


def test_a_direction_that_is_not_the_identity_survives(label_image):
    """Asserted against a rotated direction rather than the identity, because
    an image that never had one set reports the identity and a lost direction
    would look like a preserved one."""
    image = label_image()
    image.SetDirection(ROTATED_DIRECTION)

    resampled = pipeline.resample(image, _shift(1.0))

    assert resampled.GetDirection() == ROTATED_DIRECTION


@pytest.mark.parametrize("pixel_type", [
    sitk.sitkInt16, sitk.sitkUInt8, sitk.sitkFloat32, sitk.sitkFloat64,
    sitk.sitkUInt16, sitk.sitkInt32,
])
def test_the_output_keeps_the_input_s_pixel_type(label_image, pixel_type):
    """A segmentation that came in as uint8 must not leave as float: whoever
    reads it back is reading label values."""
    image = sitk.Cast(label_image(), pixel_type)

    assert pipeline.resample(image, _shift(0.3)).GetPixelID() == pixel_type


# ---------------------------------------------------------------------------
# Geometry, with a reference: the reference's grid
# ---------------------------------------------------------------------------

def test_a_reference_decides_size_spacing_origin_and_direction(label_image):
    reference = sitk.Image(3, 4, 5, sitk.sitkUInt8)
    reference.SetSpacing((2.0, 2.0, 2.0))
    reference.SetOrigin((-1.0, -2.0, -3.0))
    reference.SetDirection(ROTATED_DIRECTION)

    resampled = pipeline.resample(label_image(), _shift(1.0), reference)

    assert resampled.GetSize() == (3, 4, 5)
    assert resampled.GetSpacing() == (2.0, 2.0, 2.0)
    assert resampled.GetOrigin() == (-1.0, -2.0, -3.0)
    assert resampled.GetDirection() == ROTATED_DIRECTION


def test_a_reference_does_not_decide_the_pixel_type(label_image):
    """The reference gives the grid, not the storage: a uint8 reference must
    not truncate an int16 label map, nor a float32 scan."""
    reference = sitk.Image(4, 4, 4, sitk.sitkUInt8)

    resampled = pipeline.resample(label_image(), _shift(1.0), reference)

    assert resampled.GetPixelID() == sitk.sitkInt16


def test_the_reference_image_itself_is_not_modified(label_image):
    reference = sitk.Image(4, 4, 4, sitk.sitkUInt8)
    reference.SetSpacing((2.0, 2.0, 2.0))

    pipeline.resample(label_image(), _shift(1.0), reference)

    assert reference.GetSpacing() == (2.0, 2.0, 2.0)
    assert set(np.unique(sitk.GetArrayFromImage(reference))) == {0}


def test_a_reference_on_the_same_grid_changes_nothing_but_the_sampling(label_image):
    """A reference matching the input pins the output where the input was --
    unlike the no-reference case, where the origin follows the transform. The
    same transform therefore lands the content in two different places, which
    is the whole reason the argument exists."""
    image = label_image()
    with_reference = pipeline.resample(image, _shift(1.0), image, True)
    without = pipeline.resample(image, _shift(1.0), None, True)

    assert with_reference.GetOrigin() == (0.0, 0.0, 0.0)
    assert without.GetOrigin() == (1.0, 1.0, 1.0)
    assert not np.array_equal(sitk.GetArrayFromImage(with_reference),
                              sitk.GetArrayFromImage(without))


def test_the_content_moves_by_a_whole_voxel_when_the_shift_is_one(label_image):
    """The arithmetic end to end, on a reference grid so the origin stays put:
    a one-voxel translation moves the labels one voxel, in the direction ITK's
    resampler defines -- it maps OUTPUT points through the transform to find
    where to read, so the content moves the other way."""
    image = label_image()
    before = sitk.GetArrayFromImage(image)

    along_x = sitk.GetArrayFromImage(
        pipeline.resample(image, sitk.TranslationTransform(3, (1.0, 0.0, 0.0)),
                          image, True))
    along_z = sitk.GetArrayFromImage(
        pipeline.resample(image, sitk.TranslationTransform(3, (0.0, 0.0, 1.0)),
                          image, True))

    # A numpy view of an ITK image is indexed [z, y, x], so a shift along x
    # moves the last axis and a shift along z the first.
    assert np.array_equal(along_x[:, :, :-1], before[:, :, 1:])
    assert np.array_equal(along_z[:-1], before[1:])


def test_the_input_image_is_not_modified(label_image):
    image = label_image()
    before = sitk.GetArrayFromImage(image).copy()

    pipeline.resample(image, _shift(0.3), is_segmentation=False)

    assert np.array_equal(sitk.GetArrayFromImage(image), before)
    assert image.GetOrigin() == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# A transform that turns or flips the image
# ---------------------------------------------------------------------------
# Moving the output origin by the transform while leaving the output DIRECTION
# alone only describes the same box for a pure translation. For anything that
# turns or flips, the box lands somewhere the content is not -- and for VFACE's
# mirror it landed entirely past the far side of the head, so the resampled
# volume came back with not one non-zero voxel. AREG_CBCT then registered that
# empty volume, reported a success, and an asymmetry measurement between a
# midline point and its own mirror came out at 164 mm instead of nothing.


def _mirror():
    """VFACE's own `Matrix_mirror.tfm`: `x -> -x` about x = 0."""
    transform = sitk.AffineTransform(3)
    transform.SetMatrix((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return transform


def _quarter_turn():
    """A rotation, to say the rule is about turning and not about flipping."""
    transform = sitk.Euler3DTransform()
    transform.SetRotation(0.0, 0.0, np.pi / 2)
    return transform


def _centred_cube():
    """A cube in a box centred on the origin, like an oriented CBCT.

    Centred on purpose: that is the geometry VFACE's mirror is applied in, and
    the one where transforming the origin sends the box exactly one field of
    view away.
    """
    array = np.zeros((8, 8, 8), np.int16)
    array[2:5, 2:5, 2:5] = 7
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((-4.0, -4.0, -4.0))
    return image


def test_a_mirrored_volume_is_not_empty():
    """The defect itself, in one assertion.

    Not "the origin is right" but "there is something in the image": a test on
    the geometry alone would have passed against a box full of nothing, which is
    what a whole VFACE run was built on.
    """
    image = _centred_cube()

    mirrored = pipeline.resample(image, _mirror())

    assert sitk.GetArrayFromImage(mirrored).any(), "the mirror emptied the volume"


def test_a_mirrored_volume_occupies_the_box_it_came_from():
    """Same grid in, same grid out -- which is what the shipped reference has.

    `V_FACE/Test_Output/T2_Scan/CB/C_0001_T1_CB_Or_mir.nii.gz` spans the same x
    as the scan it was mirrored from, and holds 20 910 456 non-zero voxels
    against the original's 20 962 743. Keeping the grid reproduces that; moving
    the origin produced zero.
    """
    image = _centred_cube()

    mirrored = pipeline.resample(image, _mirror())

    assert mirrored.GetOrigin() == image.GetOrigin()
    assert mirrored.GetSize() == image.GetSize()
    assert mirrored.GetSpacing() == image.GetSpacing()


def _centroid_x(image):
    """Where the content sits on the x axis, in millimetres, not in voxels."""
    array = sitk.GetArrayFromImage(image)          # indexed (z, y, x)
    weights = array.astype(float)
    indices = np.arange(array.shape[2])
    centre = float((weights.sum(axis=(0, 1)) * indices).sum() / weights.sum())
    return image.GetOrigin()[0] + centre * image.GetSpacing()[0]


def test_a_mirrored_volume_is_actually_mirrored():
    """And it is the anatomy that moved, not the box.

    Asserted on the content's position in MILLIMETRES rather than on a reversed
    array, because the two are not the same statement: an 8-voxel grid spanning
    [-4, 4] has its voxel centres at -4 .. 3, so its own middle is at -0.5 and a
    geometric flip about x = 0 is not `array[..., ::-1]`. The physical claim is
    the one that matters -- and an implementation that returned the input
    untouched would pass the two tests above and fail this one.
    """
    image = _centred_cube()
    before = _centroid_x(image)

    after = _centroid_x(pipeline.resample(image, _mirror()))

    assert before == pytest.approx(-after, abs=1e-6)
    assert abs(before) > 0.5, "a cube on the midline could not have shown that"


def test_a_rotation_keeps_its_box_too():
    image = _centred_cube()

    turned = pipeline.resample(image, _quarter_turn())

    assert turned.GetOrigin() == image.GetOrigin()
    assert sitk.GetArrayFromImage(turned).any()


def test_a_translation_still_carries_the_origin_with_it(label_image):
    """The behaviour this narrows, asserted beside it so the contrast is read.

    A pure translation is the one case where moving the origin is
    self-consistent, it is what the port was written for, and it does not move.
    """
    image = label_image(origin=(10.0, 20.0, 30.0))

    resampled = pipeline.resample(image, sitk.TranslationTransform(3, (1.0, 2.0, 3.0)))

    assert resampled.GetOrigin() == (11.0, 22.0, 33.0)


# ---------------------------------------------------------------------------
# The question the rule asks
# ---------------------------------------------------------------------------

def test_a_translation_is_recognised_whatever_class_it_arrives_as():
    """Probed, not read off a matrix: a transform reaches this tool as any of
    half a dozen ITK classes and only some answer `GetMatrix`."""
    assert pipeline.is_pure_translation(sitk.TranslationTransform(3, (1.0, 2.0, 3.0)))

    affine = sitk.AffineTransform(3)
    affine.SetTranslation((1.0, 2.0, 3.0))
    assert pipeline.is_pure_translation(affine)

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(sitk.TranslationTransform(3, (1.0, 0.0, 0.0)))
    composite.AddTransform(sitk.TranslationTransform(3, (0.0, 2.0, 0.0)))
    assert pipeline.is_pure_translation(composite)

    assert pipeline.is_pure_translation(sitk.Euler3DTransform())  # no rotation set


def test_anything_that_turns_or_flips_is_not_one():
    assert not pipeline.is_pure_translation(_mirror())
    assert not pipeline.is_pure_translation(_quarter_turn())

    scaled = sitk.AffineTransform(3)
    scaled.Scale(2.0)
    assert not pipeline.is_pure_translation(scaled)

    # A rotation with a translation on top is still not a pure translation.
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(sitk.TranslationTransform(3, (5.0, 0.0, 0.0)))
    composite.AddTransform(_mirror())
    assert not pipeline.is_pure_translation(composite)

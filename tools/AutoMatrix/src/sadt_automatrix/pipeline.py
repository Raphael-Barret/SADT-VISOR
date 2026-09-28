"""Apply a transform to a scan, a segmentation or a landmark file.

Ported from `Automatrix_CLI/Automatrix_CLI.py`. SimpleITK reads the transform,
resamples the image and writes it back; a landmark file has its points moved by
the INVERSE transform, because a transform that maps image space maps points
the other way.

The interpolation rule is upstream's and is the important one: nearest
neighbour for a segmentation, linear otherwise. Interpolating a label map
linearly invents labels that were never in it.
"""

import json
import logging
import os

logger = logging.getLogger("AutoMatrix")

# Tokens that name what a file IS rather than whose it is, and so must not end
# up in the patient key. The shared `patient_stem` already drops the scan
# suffixes a previous ASO/AMASSS/ALI/AREG run leaves (`_Seg`, `_Or`, `_lm`...);
# these are the transform vocabulary it does not know, plus the region and jaw
# tokens that sit between the patient and the suffix.
#
# Without them `C_0001_CB_Reg_transform.tfm` keys to `C_0001_CB_Reg_transform`
# and matches no scan at all -- which is this tool's whole job.
PATIENT_TOKENS_TO_DROP = (
    "transform", "matrix", "warp", "reg", "cbreg", "mandreg", "maxreg",
    "cb", "mand", "max", "md", "mx",
)


def patient_of(filename: str) -> str:
    """The patient a file belongs to, transform or scan alike.

    Upstream derived this with fifteen chained `.split()` calls and a loop over
    `_T0` to `_T49`, so a patient genuinely named `P_Seg1` was truncated and a
    trailing `.split('.')[0]` broke any name holding a dot.
    """
    from sadt_areg_common import pairing

    return pairing.patient_stem(filename, also_drop=PATIENT_TOKENS_TO_DROP)


# What a transform may be stored as.
TRANSFORM_EXTENSIONS = (".tfm", ".mat", ".h5", ".hdf5", ".txt")

# What may be transformed. `.mrk.json` is a Slicer markups file and is handled
# as points; everything else is read as an image.
#
# Longest extension first, so a caller splitting a name against this tuple
# never cuts ".nii.gz" short at ".nii". `is_image_file` only tests membership,
# which `str.endswith` answers whatever the order, so the two uses agree.
IMAGE_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")
LANDMARK_EXTENSIONS = (".mrk.json",)


def is_landmark_file(name: str) -> bool:
    return name.lower().endswith(LANDMARK_EXTENSIONS)


def is_image_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(IMAGE_EXTENSIONS) and not is_landmark_file(name)


def is_transform_file(name: str) -> bool:
    return name.lower().endswith(TRANSFORM_EXTENSIONS)


def apply_to_landmarks(source: str, transform, destination: str) -> int:
    """Move a Slicer markups file's points. Returns how many moved.

    The INVERSE is applied, as upstream does: a transform produced by a
    registration maps image space, and a point follows the opposite way. A
    transform that cannot be inverted is an error rather than a warning and a
    file copied unchanged -- upstream logged and returned, so the output looked
    like a result.
    """
    with open(source) as handle:
        markups = json.load(handle)

    inverse = transform.GetInverse()

    moved = 0
    for group in markups.get("markups", []):
        for point in group.get("controlPoints", []):
            position = point.get("position")
            # Absent means "defined": that is the default Slicer's own markups
            # schema declares, and ASO and AREG read a control point without
            # looking at the field at all. Requiring it left a point from any
            # writer that omits it sitting at its old coordinates in a file
            # reported as transformed -- the family of failure this port exists
            # to remove.
            if point.get("positionStatus", "defined") != "defined":
                continue
            if not isinstance(position, list) or len(position) != 3:
                continue
            point["position"] = list(inverse.TransformPoint(position))
            moved += 1

    with open(destination, "w") as handle:
        json.dump(markups, handle, indent=2)
    return moved


def is_pure_translation(transform, tolerance: float = 1e-6) -> bool:
    """True when `transform` only shifts, with no rotation, scale or reflection.

    Probed rather than read off a matrix, because a transform arrives as any of
    half a dozen ITK classes -- Translation, Euler, Affine, a Composite of
    several, a bare 4x4 GreedyReg wrote -- and only some of them answer
    `GetMatrix`. What is asked here is the only property the caller cares about:
    does this transform move the three basis directions?
    """
    import numpy as np

    at_origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)))
    for axis in np.eye(3):
        moved = np.asarray(transform.TransformPoint(tuple(axis))) - at_origin
        if not np.allclose(moved, axis, atol=tolerance):
            return False
    return True


def resample(image, transform, reference=None, is_segmentation: bool = False):
    """The resampled image.

    Nearest neighbour for a segmentation, linear otherwise -- upstream's rule,
    and the one that matters: interpolating a label map linearly produces
    labels that were never in it.

    With no reference the output keeps the input's grid. Its origin also moves
    by the transform, but ONLY when the transform is a pure translation -- which
    is the case the behaviour was ported for, and the only one it is
    self-consistent in.

    **Moving the origin of a grid whose direction is left alone throws the
    output box away from the content whenever the transform turns or flips it.**
    VFACE's mirror is `x -> -x` about x = 0; on a scan spanning x in
    [-84.15, 84.15] the transformed origin is +84.15, so the output grid ran
    from +84.15 to +252.45 -- past the far side of the head -- and the resampled
    volume came out with **not one non-zero voxel**. AREG_CBCT then registered
    that empty volume onto the original, found the 168.30 mm translation that
    brings an empty box back over a head, and reported a success; the mirrored
    landmarks inherited that translation, and an asymmetry measurement between
    a midline point and its own mirror came out at 164 mm instead of nothing.

    The shipped reference output settles what the right answer is:
    `V_FACE/Test_Output/T2_Scan/CB/C_0001_T1_CB_Or_mir.nii.gz` spans the same
    x as the scan it was mirrored from and holds 20 910 456 non-zero voxels,
    which is exactly what keeping the grid produces here. So the origin move was
    never the Slicer module's behaviour for anything but a translation.
    """
    import SimpleITK as sitk

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(0)

    if reference is not None:
        resampler.SetReferenceImage(reference)
    else:
        resampler.SetSize(image.GetSize())
        resampler.SetOutputSpacing(image.GetSpacing())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(
            transform.TransformPoint(image.GetOrigin())
            if is_pure_translation(transform)
            else image.GetOrigin()
        )

    return resampler.Execute(image)


def read_transform(path: str):
    """A transform, whichever of the two shapes it was written in.

    ITK's own formats (`.tfm`, `.h5`, and the MATLAB `.mat` ITK writes) are
    read by `sitk.ReadTransform`. Greedy writes something else: a bare 4x4
    matrix, four lines of numbers, and calls it `.mat` -- so does upstream's
    own `writeIdentityInit`. `ReadTransform` refuses it with a MatlabTransformIO
    error, which is why AutoMatrix could not consume what GreedyReg produced.

    The extension does not say which it is, so the plain matrix is the
    fallback rather than a branch on the name.
    """
    import SimpleITK as sitk

    try:
        return sitk.ReadTransform(path)
    except RuntimeError as itk_error:
        matrix = _read_plain_matrix(path)
        if matrix is None:
            raise RuntimeError(
                f"{os.path.basename(path)} is neither an ITK transform nor a "
                f"4x4 matrix in text. ITK said: {itk_error}"
            ) from itk_error
        affine = sitk.AffineTransform(3)
        affine.SetMatrix([value for row in matrix[:3] for value in row[:3]])
        affine.SetTranslation([row[3] for row in matrix[:3]])
        return affine


def _read_plain_matrix(path: str):
    """A 4x4 matrix written as four lines of numbers, or None."""
    try:
        with open(path) as handle:
            rows = [
                [float(value) for value in line.split()]
                for line in handle
                if line.strip()
            ]
    except (ValueError, UnicodeDecodeError):
        return None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        return None
    return rows


# ---------------------------------------------------------------------------
# The pairing SlicerAutomatedDentalTools uses, ported verbatim.
#
# `patient_of` above drops whole underscore-delimited tokens, which is stricter
# and does not reproduce this: upstream cuts a name at the first occurrence of a
# token as a SUBSTRING, so `_Left` also truncates `_LeftMI`. Every one of the
# four datasets published with the legacy module pairs under these rules and
# under none of ours, which is the only reason they are here.
#
# The two lists are NOT the same on each side, and that asymmetry is
# load-bearing rather than an oversight: the file side leaves `_Left`/`_Right`
# alone, so `r_2_T1_LeftMI_model_Or.vtk` reaches `r_2` through its `_T1` and not
# by losing `LeftMI`. Keeping one shared list would key it to `r_2_T1`.
#
# These are a FALLBACK. Nothing calls them for a name the rule above could pair.
# ---------------------------------------------------------------------------

_LEGACY_FILE_CUTS = (
    "_Seg", "_seg", "_Scan", "_scan", "_Or", "_OR", "_MAND", "_MD", "_MAX",
    "_MX", "_CB", "_lm", "_T2", "_T1", "_Cl", "_MR",
)

_LEGACY_TRANSFORM_CUTS = (
    "_SegOr", "_Left", "_left", "_Right", "_right", "_Or", "_OR", "_MAND",
    "_MD", "_MAX", "_MX", "_CB", "_lm", "_T2", "_T1", "_Cl", "_MA", "_Mir",
    "_mir", "_Mirror", "_mirror", "_MR",
)


def _legacy_cut(name: str, cuts) -> str:
    """`name` cut at each token in turn, then at its first dot.

    Upstream chains `.split(token)[0]`, so a name that BEGINS with a token
    reduces to the empty string. That is reproduced rather than guarded here --
    the caller ignores an empty key, which is the only sane thing to pair on.
    """
    stem = name
    for token in cuts:
        stem = stem.split(token)[0]
    stem = stem.split(".")[0]
    # Upstream follows its fixed list with `_T0` .. `_T49`, which is what strips
    # a timepoint the two lists above do not already name.
    for index in range(50):
        stem = stem.split("_T" + str(index))[0]
    return stem


def legacy_file_key(filename: str) -> str:
    """The patient a scan, segmentation or landmark file belongs to, upstream's way."""
    return _legacy_cut(filename, _LEGACY_FILE_CUTS)


def legacy_transform_key(filename: str) -> str:
    """The patient a transform belongs to, upstream's way."""
    return _legacy_cut(filename, _LEGACY_TRANSFORM_CUTS)


# ---------------------------------------------------------------------------
# Scan or segmentation, read off the data instead of asked for.
#
# Which one it is decides the interpolator, and getting it wrong on a label map
# INVENTS labels that were never segmented -- so it is worth knowing rather than
# assuming. Measured on this server's own data: a mask holds 2 distinct values
# and a CBCT holds 3746 to 7993, three orders of magnitude apart, and every scan
# carries the negative Hounsfield numbers no label map ever has.
#
# The caller can still say, and a caller that says is believed: this only fills
# in for `content="Automatic"`.
# ---------------------------------------------------------------------------

# Above this many distinct values it is a scan. A label map has tens -- the
# richest here is Batch_Dental_Seg's 32 permanent teeth, 20 deciduous and 3
# structures -- while a scan normalised into 0..255 has essentially all of them,
# which is the case this number is set to separate.
LABEL_MAP_MAX_LABELS = 128

# And above this value, whatever the count. A CBCT exported unsigned lands in
# 0..4095, so it never reaches the counting step at all.
_LABEL_MAP_MAX_VALUE = 1000


def looks_like_a_label_map(image) -> bool:
    """Whether this volume holds labels rather than intensities.

    Three tests, cheapest first, each of which alone settles a scan:

    1. a floating-point volume is a scan -- labels are whole numbers;
    2. a negative value is a Hounsfield number, which no label map has;
    3. a value above `_LABEL_MAP_MAX_VALUE` is out of any label table.

    Only then are the distinct values counted, and counted in chunks so a scan
    that slipped through the first three stops at the first chunk instead of
    sorting a hundred megavoxels to learn what it already looks like.
    """
    import numpy as np
    import SimpleITK as sitk

    if image.GetPixelID() in (sitk.sitkFloat32, sitk.sitkFloat64):
        return False

    # A COPY, not GetArrayViewFromImage: a view does not keep its image alive,
    # and reading one after the image is collected quietly returns zeros.
    array = sitk.GetArrayFromImage(image)
    if array.size == 0:
        return False
    if array.min() < 0 or array.max() > _LABEL_MAP_MAX_VALUE:
        return False

    seen = set()
    for chunk in np.array_split(array.reshape(-1), 32):
        seen.update(np.unique(chunk).tolist())
        if len(seen) > LABEL_MAP_MAX_LABELS:
            return False
    return True

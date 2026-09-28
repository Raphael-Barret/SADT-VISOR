"""What AREG's schema offers: the modes, the anatomical regions it can register
on, and the tokens that name a timepoint or a jaw in a file name.

One table per concept, published to the client through `GET /tools` and read by
the engines. Nothing here is written down twice: the presentation `groups` are
derived from the same dicts the pipelines look codes up in, so a region added
below appears in the panel with no client release.
"""

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

MODALITY_CBCT = "CBCT"
MODALITY_IOS = "IOS"
# Registering an intraoral scan onto a CBCT of the same patient -- a third
# modality, not a mode of either. It was missing entirely until an audit against
# upstream found it: upstream ships AREG_Method/IOSCBCT.py (829 lines) and three
# method classes for it, and nothing on this side named it at all.
MODALITY_IOSCBCT = "IOSCBCT"

MODALITY_CHOICES = {MODALITY_CBCT: True, MODALITY_IOS: False, MODALITY_IOSCBCT: False}

AUTOMATION_SEMI = "Semi-Automated"
AUTOMATION_FULLY = "Fully-Automated"
AUTOMATION_ORIENTED = "Oriented + Fully-Automated"
# IOSCBCT only, and NOT the same thing as Semi-Automated: it takes the landmarks
# already computed on both modalities and does the cross-modality registration
# alone, predicting nothing. Upstream labels it plainly "Registration" in the
# panel, against "Semi Automated Registration" and "Fully Automated
# Registration" beside it.
AUTOMATION_REGISTRATION = "Registration"

# Fully-Automated is the default rather than the Slicer module's
# Or_Auto_CBCT: the oriented mode additionally needs an orientation reference
# bundle, and a default that cannot run without a file the user has not chosen
# yet reads as a broken tool rather than as a default.
AUTOMATION_CHOICES = {
    AUTOMATION_SEMI: False,
    AUTOMATION_FULLY: True,
    AUTOMATION_ORIENTED: False,
    AUTOMATION_REGISTRATION: False,
}

# Which automation levels each modality actually has. The schema cannot say
# "this option only exists for that modality" -- `visible_when` hides an
# argument, not one option of a choice -- so the pair is checked in
# AREGLogic.main and answered with a 422 naming what is available.
AUTOMATION_BY_MODALITY = {
    MODALITY_CBCT: (AUTOMATION_SEMI, AUTOMATION_FULLY, AUTOMATION_ORIENTED),
    MODALITY_IOS: (AUTOMATION_SEMI, AUTOMATION_FULLY),
    # No "Oriented + Fully-Automated" here: orienting before registering is a
    # CBCT step, and there is nothing to orient an intraoral scan onto in this
    # mode. Its third value is Registration instead -- landmarks in, no
    # prediction at all.
    MODALITY_IOSCBCT: (AUTOMATION_SEMI, AUTOMATION_FULLY, AUTOMATION_REGISTRATION),
}


# ---------------------------------------------------------------------------
# CBCT registration regions
# ---------------------------------------------------------------------------
# The anatomy the voxel-based registration is masked to. One run per selected
# region, each producing its own output tree -- registering on the cranial base
# and on the mandible are two different clinical questions, not two settings of
# one.

REGION_CODES = {
    "Cranial base": "CB",
    "Mandible": "MAND",
    "Maxilla": "MAX",
}

REGION_CHOICES = {name: name == "Cranial base" for name in REGION_CODES}

# The AMASSS structure whose mask each region registers against, used by the
# Fully-Automated modes to ask AMASSS for exactly what will be consumed.
# AMASSS speaks display names on its schema and codes in `segment()`; these are
# the codes (see AMASSSLogic.STRUCTURE_GROUPS["Masks"]).
REGION_MASK_STRUCTURES = {
    "CB": "CBMASK",
    "MAND": "MANDMASK",
    "MAX": "MAXMASK",
}

# The anatomical segmentations AMASSS can produce BESIDE the registration
# masks, and the second group of check boxes the original module offers
# ("AMASSS Segmentation", AREG_Method/CBCT.py's DicLandmark). Two groups
# rather than one, because they answer different questions: the regions above
# decide what the registration is masked to, these decide what comes back for
# the clinician to look at. Nothing here changes the registration.
#
# The six are AMASSS's non-mask structures, which is exactly what the original
# offers. Spelled the way this repository spells the regions above ("Cranial
# base", not "Cranial Base"): one schema, one convention.
SEGMENTATION_CODES = {
    "Cranial base": "CB",
    "Cervical vertebra": "CV",
    "Mandible": "MAND",
    "Maxilla": "MAX",
    "Skin": "SKIN",
    "Upper airway": "UAW",
}

# None ticked: a registration run returns a registration. Asking for skin and
# airway on every patient would add minutes of GPU and gigabytes of output to
# a run nobody asked to segment.
SEGMENTATION_CHOICES = {name: False for name in SEGMENTATION_CODES}

# Tokens that name a region inside a mask's file name, matched as WHOLE tokens
# of the stem rather than as substrings.
#
# The original matched substrings (`"cb" in basename.lower()`), which makes
# every file whose name contains "CBCT" a cranial-base mask -- and "max" also
# matches a patient named MAX_01, "md" matches almost anything. Token matching
# is what makes `P1_CBCT_seg.nii.gz` not a cranial-base mask.
REGION_TOKENS = {
    "CB": ("cb", "cbmask", "cranialbase", "cranial"),
    "MAND": ("mand", "md", "mandmask", "mandible"),
    "MAX": ("max", "mx", "maxmask", "maxilla"),
}

# A file naming itself a segmentation. Combined with the region tokens above:
# a mask has to say BOTH what it is and which structure it covers.
MASK_TOKENS = ("mask", "seg", "pred", "segmentation")


def region_code(name: str) -> str:
    return REGION_CODES[name]


def region_name(code: str) -> str:
    """The display name of a region code ('CB' -> 'Cranial base')."""
    for name, value in REGION_CODES.items():
        if value == code:
            return name
    return code


# ---------------------------------------------------------------------------
# Timepoints
# ---------------------------------------------------------------------------
# T1 and T2 arrive in two separate folders, so unlike ASO -- where both
# timepoints of a subject sit side by side and stripping "_T1" would merge
# them -- the token here IS the timepoint and stripping it is what pairs the
# two folders.

TIMEPOINT_TOKENS = ("t1", "t2", "t0")

# Suffixes a previous run of AREG, ASO, AMASSS or ALI leaves on a name.
# Longest first, so "_Scanreg" is not cut short by "_Scan".
#
# Matched as WHOLE TOKENS and truncated from, never as a substring at any index:
# "_Seg" is a mark a previous run left, "_Seg1" is part of somebody's name. See
# `pairing._token_aligned_index`, and note that this table is case-SENSITIVE --
# which is why "_Scan"/"_scan", "_Seg"/"_seg" and "_Or"/"_OR" are each listed
# twice and "_SEG" is not listed at all.
PATIENT_SUFFIXES = (
    "_lm_Pred", "_Scanreg", "_MERGED", "_OutReg", "_SegOr",
    "_scan", "_Scan", "_Seg", "_seg", "_Or", "_OR", "_lm",
)

# Which of those say "this scan was ORIENTED", and so may carry the frame it
# was oriented into just before them. ASO appends no bare marker: its caller
# names the suffix, and VFACE names it `CB_Or` and `MAX_Or`, so the frame ends
# up inside the name -- `C_0002_T1_CB_Or.nii.gz`. See
# `pairing._frame_aligned_index` for why a subject's key must not keep it.
ORIENTATION_SUFFIXES = ("_SegOr", "_Or", "_OR")


# ---------------------------------------------------------------------------
# Jaws (IOS)
# ---------------------------------------------------------------------------
# Same table and the same rule as ASO's: a mesh whose name does not say which
# jaw it is gets refused rather than defaulted. `AREG_IOS_utils.Sort` defaulted
# every non-Upper file to Lower, so a maxillary mesh named `patient1.vtk` was
# registered as a mandible and returned as a success.

JAW_UPPER = "Upper"
JAW_LOWER = "Lower"

JAW_TOKENS = {
    "u": JAW_UPPER, "up": JAW_UPPER, "upper": JAW_UPPER,
    "maxilla": JAW_UPPER, "max": JAW_UPPER, "mx": JAW_UPPER,
    "l": JAW_LOWER, "low": JAW_LOWER, "lower": JAW_LOWER,
    "mandible": JAW_LOWER, "mandibule": JAW_LOWER, "mand": JAW_LOWER, "md": JAW_LOWER,
}


# ---------------------------------------------------------------------------
# IOS registration patch
# ---------------------------------------------------------------------------
# Which stable region the two timepoints are aligned on. The two are different
# arches, not two settings of one thing: the palate exists only on the maxilla
# and the mucogingival line only matters on the mandible, so picking one also
# picks which arch is registered and which is carried along.

PATCH_PALATE = "Palate (upper arch)"
PATCH_MGL = "Mucogingival line (lower arch)"

PATCH_CHOICES = {PATCH_PALATE: True, PATCH_MGL: False}

# The jaw each patch registers.
PATCH_JAW = {PATCH_PALATE: JAW_UPPER, PATCH_MGL: JAW_LOWER}

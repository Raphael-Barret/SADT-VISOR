"""The vocabularies VFACE's arguments are drawn from, and the tables behind them.

Three axes, and they are upstream's own combo boxes:

    mode      how far down the pipeline the caller's scans already are
    study     what the second timepoint IS -- the patient's own mirror, or a
              real follow-up scan
    outputs   measurements and a classification, heat maps, or both

**`study` is the one worth stopping on.** In an asymmetry assessment there is
no second scan: the "T2" is the patient's own scan MIRRORED, and the whole
pipeline then measures a patient against themselves. A longitudinal study is
the ordinary reading -- a real follow-up, `t2`, compared with the baseline. The
steps are otherwise the same, which is why one tool does both and why the
mirroring is where they part.

One spelling is corrected from upstream: `Asymmetry Assesment` is a typo, and
the published options are what a client sends. Fixing it now costs nothing --
no client holds the old spelling, this tool has never been served -- and
leaving it would have made the typo an API.
"""

# --- mode ------------------------------------------------------------------

MODE_FULL = "Full pipeline"
MODE_ORIENTED = "File already Oriented"
MODE_REGISTERED = "File already Registered"
MODES = (MODE_FULL, MODE_ORIENTED, MODE_REGISTERED)

# --- study -----------------------------------------------------------------

STUDY_ASYMMETRY = "Asymmetry assessment"
STUDY_LONGITUDINAL = "Longitudinal study"
STUDIES = (STUDY_ASYMMETRY, STUDY_LONGITUDINAL)

# --- outputs ---------------------------------------------------------------

OUTPUT_BOTH = "Measurements and heat maps"
OUTPUT_QUANTITATIVE = "Measurements and classification"
OUTPUT_VISUALISATION = "Heat maps"
OUTPUTS = (OUTPUT_BOTH, OUTPUT_QUANTITATIVE, OUTPUT_VISUALISATION)


def wants_measurements(outputs: str) -> bool:
    return outputs in (OUTPUT_BOTH, OUTPUT_QUANTITATIVE)


def wants_heat_maps(outputs: str) -> bool:
    return outputs in (OUTPUT_BOTH, OUTPUT_VISUALISATION)


# --- the two frames a scan is oriented into --------------------------------

# A scan is oriented TWICE, into two different frames, and everything after
# that is done in one or the other. The cranial base frame is the Frankfort
# horizontal and the mid-sagittal plane; the maxillary frame is the occlusal
# and mid-sagittal plane. A measurement of the mandible against the cranial
# base and one of the maxilla against the occlusal plane are different
# questions, and each is asked in the frame it is defined in.
FRAME_CRANIAL_BASE = "CB"
FRAME_MAXILLA = "MAX"

FRAMES = {
    FRAME_CRANIAL_BASE: {
        # The subfolder of the orientation reference bundle. Upstream joins
        # these names onto `gold_folder`, and the published bundle carries both.
        "reference": "Frankfurt Horizontal and Midsagittal Plane",
        "landmarks": ["Ba", "LPo", "N", "RPo", "S", "LOr", "ROr"],
        "suffix": "CB_Or",
    },
    FRAME_MAXILLA: {
        "reference": "Occlusal and Midsagittal Plane",
        "landmarks": ["ANS", "IF", "PNS", "UL6O", "UR1O", "UR6O"],
        "suffix": "MAX_Or",
    },
}

# --- the three regions a measurement is made on ----------------------------

REGION_CRANIAL_BASE = "Cranial base"
REGION_MANDIBLE = "Mandible"
REGION_MAXILLA = "Maxilla"
REGIONS = (REGION_CRANIAL_BASE, REGION_MANDIBLE, REGION_MAXILLA)

# Which frame each region is worked in, which AMASSS structure masks it, and
# which AREG region registers on it.
#
# The mandible is worked in the CRANIAL BASE frame, not one of its own, and
# that is the point: a mandible is measured by how it sits relative to the
# skull, so the frame has to be the one the skull defines. Only the maxilla
# gets the occlusal frame.
#
# **The structure codes are the MASK variants, and the difference is not
# cosmetic.** AMASSS publishes `CB`/`MAND`/`MAX` -- the anatomical
# segmentations, which follow the bone -- and `CBMASK`/`MANDMASK`/`MAXMASK`,
# which are the regions a registration is confined to. Upstream asks for the
# second set (`TranslateModels(..., mask=True)`), and so does AREG. Sending the
# first would hand the registration a segmentation where it expects a mask: it
# would run, produce a transform, and report success on the wrong anatomy.
# `code` is the three-letter name the outputs are named by -- the measurement
# workbook is `Measurements_CB.xlsx` and every feature column starts with it --
# so it is what the panel shows beside each option: a reader ticking a region
# should be able to tell which file is going to carry the answer.
REGION_TABLE = {
    REGION_CRANIAL_BASE: {
        "frame": FRAME_CRANIAL_BASE, "structure": "CBMASK", "areg": "Cranial base",
        "code": "CB",
    },
    REGION_MANDIBLE: {
        "frame": FRAME_CRANIAL_BASE, "structure": "MANDMASK", "areg": "Mandible",
        "code": "MAND",
    },
    REGION_MAXILLA: {
        "frame": FRAME_MAXILLA, "structure": "MAXMASK", "areg": "Maxilla",
        "code": "MAX",
    },
}

# {display name: code}, derived so the table above stays the only place a region
# is described. `dispatch._short` and the panel's `option_help` both read this.
REGION_CODES = {name: entry["code"] for name, entry in REGION_TABLE.items()}


def frames_needed(regions) -> list:
    """The frames a set of regions has to be oriented into, in a stable order.

    Derived, never listed: a region added to `REGION_TABLE` and forgotten here
    would be registered against a scan nothing had oriented.
    """
    wanted = {REGION_TABLE[region]["frame"] for region in regions}
    return [frame for frame in (FRAME_CRANIAL_BASE, FRAME_MAXILLA) if frame in wanted]


def structures_for(frame: str, regions) -> list:
    """The AMASSS structures to segment on the scans oriented into `frame`.

    One call per frame rather than one per region: AMASSS loads a network per
    structure and a cohort per call, so asking for both of the cranial base
    frame's structures at once is one pass over the scans instead of two.
    """
    return [
        REGION_TABLE[region]["structure"]
        for region in REGIONS
        if region in regions and REGION_TABLE[region]["frame"] == frame
    ]

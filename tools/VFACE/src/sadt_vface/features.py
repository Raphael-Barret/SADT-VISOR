"""From measurement tables to the feature row a classifier reads.

Two steps, both upstream's:

* **`to_stats`** turns each measurement row into SIGNED numbers. The tables
  `aq3dc` writes carry magnitudes and a label -- `2.4 mm, Medial` -- because
  that is what a clinician reads. A model cannot read a label, so the label is
  what puts the sign back on, and which label means "negative" depends on
  whether the row is dental or skeletal and, for a tooth, on which segment of
  which arch it sits in.
* **`build_feature_table`** reads a TEMPLATE workbook whose column names encode
  what each feature is -- `MAND_RCo_LCo_RL` is the transverse component of the
  distance between the two condyles, measured on the mandible -- and fills one
  row per patient by looking each of them up in the right region's table.

The template is the contract with the model: the classifier selects the columns
named in its own `feature_name_`, so a feature the template does not carry is a
column the model will not find. It is supplied by the caller, alongside the
measurement lists, because it belongs with the model that was trained on it.
"""

import logging
import os
import re

from .errors import ToolInputError

logger = logging.getLogger(__name__)

# The teeth a `Landmarks` cell can name. Upstream keeps the LAST match rather
# than the first, which matters for a measurement naming two of them.
TEETH = (
    "UR8", "UR7", "UR6", "UR5", "UR4", "UR3", "UR1", "UR2",
    "UL8", "UL7", "UL6", "UL5", "UL4", "UL3", "UL1", "UL2",
    "LR8", "LR7", "LR6", "LR5", "LR4", "LR3", "LR1", "LR2",
    "LL8", "LL7", "LL6", "LL5", "LL4", "LL3", "LL1", "LL2",
)

# The columns a stats table can carry, in upstream's order. One that comes out
# empty, or holds nothing but `x`, is dropped: it is a question none of these
# measurements answered.
STAT_COLUMNS = (
    "ID", "Landmarks", "Transverse", "AP", "Vertical", "3D",
    "Yaw", "Pitch", "Roll", "BL", "MD", "Rotation", "Arch", "Segment",
)

NOT_APPLICABLE = "x"

# What a feature column's name says. A column is `<REGION>_<landmarks>_<AXIS>`,
# and these two tables are what take it apart.
REGIONS = ("CB", "MAND", "MAX")
AXES = {
    "RL": "Transverse",
    "IS": "Vertical",
    "AP": "AP",
    "Pitch": "Pitch",
    "Yaw": "Yaw",
    "Roll": "Roll",
}

# Which axes name a distance -- written `A - B` in the measurement table -- and
# which name an angle, written `A-B / C-D`. The separator is how a feature
# column finds its row.
_DISTANCE_AXES = ("Transverse", "Vertical", "AP")


def _identifier(patient: str) -> str:
    """The ID the feature table is keyed by.

    A `P`/`Pat`/`Patient` prefix is stripped only when it is glued to a number,
    which is what this was for. Chopping the first character unconditionally
    turned `P_0001` into `_0001` and `P` into the empty string -- and an empty
    ID leaves the post-processing with nothing to group its rows by.
    """
    numbered = re.fullmatch(r"(?:patient|pat|p)[ _-]?(\d+)", str(patient), re.IGNORECASE)
    return numbered.group(1) if numbered else str(patient)


def _tooth_in(label: str):
    """The tooth a `Landmarks` cell names, or None. The LAST match, as upstream."""
    found = None
    for tooth in TEETH:
        if tooth in label:
            found = tooth
    return found


def _signed(value, meaning, negative_when):
    """A magnitude and its label, back into one signed number.

    `x` passes through untouched: it is not a small measurement, it is a
    question this row does not answer, and the post-processing reads it as
    "leave that feature empty".
    """
    if value == NOT_APPLICABLE or value == "":
        return NOT_APPLICABLE
    value = float(value)
    return -value if meaning in negative_when else value


def to_stats(rows) -> dict:
    """Signed features, one entry per measurement row, in columns of lists.

    Dental and skeletal rows are read with different tables, and the dental
    ones again by segment: on an anterior tooth the pitch is the buccolingual
    inclination and the roll is the mesiodistal one, and on a posterior tooth
    they swap. That is not a convention this port chose -- it is the anatomy,
    and it is why the two branches are not one.
    """
    stats = {name: [] for name in STAT_COLUMNS}

    for row in rows:
        stats["ID"].append(_identifier(row["Patient"]))
        stats["Landmarks"].append(row["Landmarks"])

        tooth = _tooth_in(str(row["Landmarks"]))
        if tooth is not None:
            stats["Arch"].append(0 if "U" in tooth else 1)
            anterior = "1" in tooth or "2" in tooth
            stats["Segment"].append(1 if anterior else 0)

            if anterior:
                stats["AP"].append(str(_signed(row["A-P Component"], row["A-P Meaning"], ("L",))))
                stats["Transverse"].append(
                    str(_signed(row["R-L Component"], row["R-L Meaning"], ("D",))))
                stats["BL"].append(
                    str(_signed(row["Pitch Component"], row["Pitch Meaning"], ("L",))))
                stats["MD"].append(
                    str(_signed(row["Roll Component"], row["Roll Meaning"], ("D",))))
            else:
                stats["AP"].append(str(_signed(row["A-P Component"], row["A-P Meaning"], ("D",))))
                stats["Transverse"].append(
                    str(_signed(row["R-L Component"], row["R-L Meaning"], ("B",))))
                stats["MD"].append(
                    str(_signed(row["Pitch Component"], row["Pitch Meaning"], ("D",))))
                stats["BL"].append(
                    str(_signed(row["Roll Component"], row["Roll Meaning"], ("L",))))

            stats["Vertical"].append(
                str(_signed(row["S-I Component"], row["S-I Meaning"], ("I",))))
            stats["Rotation"].append(
                str(_signed(row["Yaw Component"], row["Yaw Meaning"], ("DR",))))
            stats["3D"].append(str(row["3D Distance"]))
            stats["Yaw"].append(NOT_APPLICABLE)
            stats["Pitch"].append(NOT_APPLICABLE)
            stats["Roll"].append(NOT_APPLICABLE)
        else:
            stats["Arch"].append(NOT_APPLICABLE)
            stats["Segment"].append(NOT_APPLICABLE)
            stats["BL"].append(NOT_APPLICABLE)
            stats["MD"].append(NOT_APPLICABLE)
            stats["Rotation"].append(NOT_APPLICABLE)

            stats["Transverse"].append(
                str(_signed(row["R-L Component"], row["R-L Meaning"], ("Medial", "L"))))
            stats["AP"].append(str(_signed(row["A-P Component"], row["A-P Meaning"], ("P",))))
            stats["Vertical"].append(
                str(_signed(row["S-I Component"], row["S-I Meaning"], ("S",))))
            stats["Yaw"].append(str(_signed(row["Yaw Component"], row["Yaw Meaning"], ("CounterC",))))
            stats["Pitch"].append(
                str(_signed(row["Pitch Component"], row["Pitch Meaning"], ("CounterC",))))
            stats["Roll"].append(
                str(_signed(row["Roll Component"], row["Roll Meaning"], ("CounterC",))))
            stats["3D"].append(str(row["3D Distance"]))

    return {
        name: values for name, values in stats.items()
        if values and not all(str(value) == NOT_APPLICABLE for value in values)
    }


# ---------------------------------------------------------------------------
# The feature template
# ---------------------------------------------------------------------------

def describe_feature(column: str) -> dict:
    """What a feature column's NAME says it is.

    `MAND_RCo_LCo_RL` is the transverse component of the distance between the
    two condyles, on the mandible. A column naming more than two landmarks is
    an AVERAGE over the pairs it names -- `CB_RPo_LPo_ROr_LOr_IS` is the mean
    of two vertical measurements -- and one with a `/` in it names lines rather
    than points, which is an angle.
    """
    described = {"region": None, "axis": None, "average": False, "landmarks": []}

    remaining = column
    for region in REGIONS:
        if region in column:
            described["region"] = region
            remaining = remaining.replace(region + "_", "")
    for suffix, axis in AXES.items():
        if suffix in column:
            described["axis"] = axis
            remaining = remaining.replace("_" + suffix, "")

    if "/" not in remaining:
        parts = remaining.split("_")
        described["landmarks"] = parts
        described["average"] = len(parts) > 2
    else:
        # An angle: `A-B / C-D`, and an average of angles is
        # `A-B / C-D_E-F / G-H` once the separators are flattened.
        parts = remaining.split("/")
        flattened = []
        for index, part in enumerate(parts):
            if index % 2 == 0:
                flattened.append(part)
            else:
                halves = part.split("_")
                if len(halves) == 4:
                    flattened.append(halves[0] + "_" + halves[1])
                    flattened.append(halves[2] + "_" + halves[3])
                else:
                    flattened.append(part)
        described["landmarks"] = flattened
        described["average"] = len(parts) > 2
    return described


def _label_for(described: dict, pair: int) -> str:
    """The `Landmarks` cell a feature column refers to, for one of its pairs."""
    landmarks = described["landmarks"]
    first, second = landmarks[2 * pair], landmarks[2 * pair + 1]
    if described["axis"] in _DISTANCE_AXES:
        return f"{first} - {second}"
    return f"{first} / {second}".replace("_", "-")


def _pairs_in(described: dict) -> int:
    return max(len(described["landmarks"]) // 2, 1)


def _index_by_patient(stats: dict) -> dict:
    """`{ID: {Landmarks: row}}`, keeping the first row of a repeated landmark."""
    index = {}
    rows = [
        dict(zip(stats.keys(), values))
        for values in zip(*stats.values())
    ] if stats else []
    for row in rows:
        index.setdefault(row["ID"], {}).setdefault(row["Landmarks"], row)
    return index


def build_feature_table(by_region: dict, template_columns, report: dict = None) -> list:
    """One row per patient, with the template's columns filled from the tables.

    `by_region` maps `CB`/`MAND`/`MAX` to that region's stats. A column naming
    a landmark the run never produced is left EMPTY rather than ending the
    build: upstream indexed it blind, so one missing landmark raised a KeyError,
    the workbook was never written, and every later step failed looking for a
    file that did not exist. A column left empty is visible and recoverable; a
    run stopped halfway is not.
    """
    indexed = {region: _index_by_patient(stats) for region, stats in by_region.items()}
    described = {column: describe_feature(column) for column in template_columns}

    patients = sorted({patient for index in indexed.values() for patient in index})
    if not patients:
        raise ToolInputError(
            "No patient has any measurement, so there is nothing to classify."
        )

    records = []
    for patient in patients:
        record = {"ID": patient}
        for column, description in described.items():
            region = description["region"]
            by_landmark = indexed.get(region, {}).get(patient)
            if by_landmark is None:
                # A column whose whole REGION was never measured, because the
                # run was not asked to register it. Reported like any other
                # empty column, and the region named once beside it: "the
                # template asks for the maxilla and this run did not measure
                # it" is the sentence that tells a reader what to change, and
                # the silence here is what made a classification that could
                # feed no model hard to read.
                _note_empty(report, patient, column)
                if report is not None and region:
                    absent = report.setdefault("features_regions_absent", [])
                    if region not in absent:
                        absent.append(region)
                continue

            values = []
            for pair in range(_pairs_in(description)):
                value = _value_of(by_landmark, description, pair)
                if value is None:
                    # Averaging what is left would quietly report a different
                    # measurement than the column claims to be.
                    values = None
                    break
                values.append(value)
            if not values:
                if values is None:
                    _note_empty(report, patient, column)
                continue
            record[column] = sum(values) / len(values)
        records.append(record)
    return records


def _note_empty(report, patient: str, column: str) -> None:
    """Say that one feature column came out empty for one patient."""
    if report is None:
        return
    report.setdefault("features_empty", {}).setdefault(patient, []).append(column)


def _value_of(by_landmark: dict, described: dict, pair: int):
    label = _label_for(described, pair)
    row = by_landmark.get(label)
    if row is None:
        return None
    try:
        return float(row[described["axis"]])
    except (KeyError, TypeError, ValueError):
        return None


def read_template_columns(path: str) -> list:
    """The feature columns a template workbook asks for.

    `ID` and the three answer columns are dropped: they are what the classifier
    WRITES, and a model asked to read its own answer as a feature would be
    reading the training set's labels.
    """
    import pandas as pd

    try:
        template = pd.read_excel(path)
    except Exception as exc:  # noqa: BLE001 - pandas raises a family of these
        raise ToolInputError(
            f"'{os.path.basename(path)}' could not be read as a feature template: {exc}"
        ) from exc

    answers = ("ID", "Asymmetry", "Mand", "Max")
    columns = [column for column in template.columns if column not in answers]
    if not columns:
        raise ToolInputError(
            f"'{os.path.basename(path)}' names no feature column. It should carry one "
            "column per feature the model was trained on, beside ID, Asymmetry, Mand "
            "and Max."
        )
    return columns


def write_table(records, columns, path: str) -> str:
    import pandas as pd

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(records, columns=list(columns)).to_excel(path, index=False)
    return path

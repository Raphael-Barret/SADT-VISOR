"""Pairing a T1 folder with a T2 folder, and finding the masks that go with T1.

AREG's whole input is two timepoints of the same subjects, sent as two folders,
and everything downstream keys off the pairing this module produces. One
implementation shared by both engines: `AREG_CBCT_utils.GetPatients` and the
near-identical copy in `AREG_Method/CBCT.py` had drifted into two signatures
with two different mask rules.

Three defects of those copies are fixed by construction:

* the patient key was a BASE NAME, so `scanT1.nii.gz` in two subfolders became
  one patient, in the working dict and again in the flat output folder. The key
  is the path relative to the input root, and the output mirrors that tree;
* `.split(".")[0]` truncated a name at its first dot, so `P1.2_scan.nii.gz`
  became patient `P1`. Extensions are split off properly, compound ones
  included;
* mask regions were matched as SUBSTRINGS: `"cb" in basename.lower()` makes
  every file whose name contains CBCT a cranial-base mask, `"max"` matches a
  patient named MAX_01 and `"md"` matches almost anything. Matching is on whole
  tokens of the stem.

A fourth was this module's own, and had the same shape. `PATIENT_SUFFIXES` were
matched with `stem.find(suffix)` at any index, so `P_Seg1_T1.nii.gz` and
`P_Seg2_T1.nii.gz` both keyed to `P` -- two subjects collapsed into one, one of
them silently lost or given the other's transform -- and any identifier merely
CONTAINING a suffix was truncated at it (`SMITH_ORTHO` to `SMITH`). A suffix now
has to end on a token boundary; see `_token_aligned_index`.
"""

import os
import re

from . import catalogs
from .errors import ToolInputError

SCAN_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")

# See AMASSSLogic.compressed_extension: NIfTI and GIPL take an external .gz,
# NRRD compresses inside the file and ITK has no ".nrrd.gz" writer at all.
_COMPRESSED_EXTENSIONS = {".nii": ".nii.gz", ".gipl": ".gipl.gz", ".nrrd.gz": ".nrrd"}

_SEPARATORS = re.compile(r"([_\-.\s]+)")
# One separator character, for testing whether an index falls on a token
# boundary. `_SEPARATORS` matches a whole run and is what SPLITS a stem;
# this is what asks "is the character on this side of a match a separator".
_SEPARATOR = re.compile(r"[_\-.\s]")

# Every region's own tokens, in one set. A mask's key drops them, so that
# `P1_MAND_seg.nii.gz` keys to the `P1` its scan keys to rather than `P1_MAND`;
# an oriented scan's key drops the one an orientation suffix follows, for the
# reason `_frame_aligned_index` gives.
ANATOMY_TOKENS = frozenset(
    token for group in catalogs.REGION_TOKENS.values() for token in group
)


def split_scan_extension(filename: str) -> tuple:
    """('scan.nii.gz') -> ('scan', '.nii.gz'), compound extensions preserved."""
    lower = filename.lower()
    for extension in SCAN_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)], filename[-len(extension):]
    return os.path.splitext(filename)


def compressed_extension(extension: str) -> str:
    return _COMPRESSED_EXTENSIONS.get(extension.lower(), extension)


def is_scan_file(filename: str) -> bool:
    return filename.lower().endswith(SCAN_EXTENSIONS)


def _split_jaw_timepoint(part: str) -> list:
    """`UpperT1` -> `['Upper', 'T1']`. One token in, one or two out.

    Upstream's own AREG test set is named `A2_UpperT1.vtk` / `A2_UpperT2.vtk`:
    the jaw and the timepoint run together with no separator, so the whole thing
    is a single token, `uppert1`, matching neither the jaw table nor the
    timepoint one. Two consequences, both silent until a run failed:
    `patient_stem` dropped nothing and made the two timepoints two patients, and
    `jaw_of` found no jaw at all.

    Deliberately narrow, and keyed on the two STATIC tables rather than on what
    a caller asked to drop: the split fires only when the prefix is a known jaw
    token AND the suffix is a known timepoint. `PAT1` is therefore untouched
    (`pa` is not a jaw), and so is any identifier that merely ends in something
    timepoint-shaped. Splitting on every camelCase boundary would start eating
    patient identifiers, which is the failure this module exists to prevent.
    """
    lowered = part.lower()
    for timepoint in catalogs.TIMEPOINT_TOKENS:
        if not lowered.endswith(timepoint) or len(lowered) <= len(timepoint):
            continue
        if lowered[: -len(timepoint)] in catalogs.JAW_TOKENS:
            cut = len(lowered) - len(timepoint)
            return [part[:cut], part[cut:]]
    return [part]


def split_parts(stem: str) -> list:
    """`_SEPARATORS.split`, plus the concatenated jaw+timepoint split.

    Separators AND their surroundings, so `_drop_tokens` can rebuild the stem
    from what it keeps. `tokens()` is the same thing without the separators.
    """
    out = []
    for part in _SEPARATORS.split(stem):
        if part and not _SEPARATORS.fullmatch(part):
            out.extend(_split_jaw_timepoint(part))
        else:
            out.append(part)
    return out


def tokens(stem: str) -> tuple:
    """The lowercase words of a file stem, split on _ - . and whitespace.

    Token matching rather than substring matching is the whole point: see this
    module's docstring for what `"cb" in "P1_CBCT_seg"` used to cost.
    """
    return tuple(
        part.lower()
        for part in split_parts(stem)
        if part and not _SEPARATORS.fullmatch(part)
    )


def has_token(stem: str, wanted) -> bool:
    present = set(tokens(stem))
    return any(token in present for token in wanted)


def _drop_tokens(stem: str, unwanted) -> str:
    """`stem` without the given tokens, separators collapsed.

    Case is preserved for what survives: the key ends up in output paths, and a
    patient folder should keep the name its owner gave it.
    """
    parts = split_parts(stem)
    kept = [
        part
        for part in parts
        if not (part and not _SEPARATORS.fullmatch(part) and part.lower() in unwanted)
    ]
    return re.sub(r"[_\-.\s]+", "_", "".join(kept)).strip("_-. ")


def _token_aligned_index(stem: str, suffix: str) -> int:
    """Where `suffix` occurs in `stem` as WHOLE tokens, or -1.

    `str.find` alone was the defect this exists to remove. Every entry of
    `catalogs.PATIENT_SUFFIXES` opens on an underscore, so a raw `find` is
    already anchored on its left -- and anchored nowhere on its right, which is
    what let `_Seg` match inside `_Seg1` and `_scan` inside `_scanned`:

    * `P_Seg1_T1.nii.gz` and `P_Seg2_T1.nii.gz` both truncated to `P`, so two
      subjects became one patient and one of them was silently dropped, or
      given the other's transform;
    * any identifier merely CONTAINING a suffix was cut at it -- `SMITH_ORTHO`
      to `SMITH`, `A_Segmentation` to `A`, `P1_Orion` to `P1`.

    A match therefore has to end on a token boundary: the end of the stem, or a
    separator. The left side is checked too rather than assumed, so a suffix
    that does not start with a separator would still be matched as a token
    rather than as a substring.
    """
    start = stem.find(suffix)
    while start >= 0:
        end = start + len(suffix)
        opens_on_a_boundary = (
            start == 0
            or bool(_SEPARATOR.match(suffix[0]))
            or bool(_SEPARATOR.match(stem[start - 1]))
        )
        closes_on_a_boundary = end == len(stem) or bool(_SEPARATOR.match(stem[end]))
        if opens_on_a_boundary and closes_on_a_boundary:
            return start
        start = stem.find(suffix, start + 1)
    return -1


def _frame_aligned_index(stem: str, index: int) -> int:
    """`index` moved left over the ORIENTATION FRAME the suffix was added to.

    ASO appends no bare `_Or`: its caller names the suffix, and names the frame
    in it. VFACE asks for `CB_Or` and `MAX_Or`, so a cohort comes back as
    `C_0002_T1_CB_Or.nii.gz` -- "C_0002, seen in the cranial base frame".

    A frame is a point of view on a subject, not the subject, and every other
    identity rule in this family already reads it that way:
    `AutoMatrix.PATIENT_TOKENS_TO_DROP` drops `cb`/`mand`/`max` from a patient
    key, `sadt_vface.landmarks.DECORATION_TOKENS` stops the identifier at the
    first of them, and `discover_masks` drops them from a mask's name. Only
    this function's caller kept it, and that made a scan and the mask AMASSS
    produced FROM THAT SAME SCAN two different subjects -- `C_0002_CB` against
    `C_0002`. A VFACE run then failed with "no Cranial base mask for this
    subject" after paying for an orientation, a segmentation and three
    registrations.

    Narrow on purpose: the ONE token immediately before the suffix, only when it
    is a region word, and only for a suffix that says an orientation happened.
    `C_0001_T1_Or.nii.gz` -- AREG's own oriented mode, which names no frame --
    is untouched, and so is a subject whose identifier merely holds a region
    word somewhere else.
    """
    head = stem[:index]
    parts = split_parts(head)
    offset = len(head)
    for part in reversed(parts):
        offset -= len(part)
        if not part or _SEPARATORS.fullmatch(part):
            continue
        if part.lower() not in ANATOMY_TOKENS:
            return index
        while offset > 0 and _SEPARATOR.match(head[offset - 1]):
            offset -= 1
        # Never the whole stem: a file called `CB_Or.nii.gz` says nothing about
        # a subject, and an empty key would collapse every such file into one.
        return offset if offset > 0 else index
    return index


def patient_stem(filename: str, also_drop=(), drop_timepoint: bool = True) -> str:
    """The subject a file belongs to, from its name alone.

    Strips the extension, any suffix a previous ASO/AMASSS/ALI/AREG run left,
    and the timepoint token -- which is what pairs `P1_T1_scan.nii.gz` in the
    T1 folder with `P1_T2.nii.gz` in the T2 folder. `also_drop` adds tokens to
    remove, used for masks so `P1_T1_MAND_seg.nii.gz` keys to the same patient
    as the scan it belongs to.

    Unlike ASO's equivalent, the timepoint token IS stripped here, and for the
    opposite reason: there both timepoints of a subject sit in one folder and
    are two separate scans to orient, so collapsing them lost one. Here they
    arrive in two folders and collapsing them is the pairing.

    `drop_timepoint=False` keeps it, which identifies ONE SCAN rather than a
    subject. That is what the mucogingival landmarks need: there is one landmark
    file per scan, both timepoints' files routinely sit in one folder, and
    collapsing them would make a patient's two files indistinguishable.
    """
    stem, extension = split_scan_extension(filename)
    if extension.lower() not in SCAN_EXTENSIONS:
        stem = os.path.splitext(filename)[0]
        # .mrk.json and friends: strip the inner extension too.
        if stem.lower().endswith(".mrk"):
            stem = stem[: -len(".mrk")]

    for suffix in catalogs.PATIENT_SUFFIXES:
        # Truncation, not deletion: everything the suffix introduces goes with
        # it, so `A1_seg_CBMASK.nii.gz` keys to `A1` and not to `A1_CBMASK`.
        index = _token_aligned_index(stem, suffix)
        if index > 0:
            if suffix in catalogs.ORIENTATION_SUFFIXES:
                # And the frame it was oriented INTO goes with it too.
                index = _frame_aligned_index(stem, index)
            stem = stem[:index]

    unwanted = set(also_drop)
    if drop_timepoint:
        unwanted |= set(catalogs.TIMEPOINT_TOKENS)
    return _drop_tokens(stem, unwanted)


def is_previous_output(filename: str, suffix: str) -> bool:
    """True if this file looks like something a previous AREG run wrote.

    Running twice on the same folder must not register an already-registered
    scan: `P1_CB_Reg.nii.gz` sorts before `P1_scan.nii.gz`, so without this the
    second run would silently take the first run's output as its input.

    Matched on a whole trailing token, not with `suffix in stem`: with the
    default suffix "Reg" that substring test would also exclude a patient
    called `Regina`.
    """
    if not suffix:
        return False
    stem, _ = split_scan_extension(filename)
    stem = stem[: -len("_transform")] if stem.endswith("_transform") else stem
    return stem.endswith(f"_{suffix}")


def discover(root: str, suffix: str, accept=is_scan_file) -> dict:
    """{patient key: path} for one timepoint folder.

    The key is `<relative directory>/<patient stem>`. A patient with several
    matching files in one directory keeps the first in sorted order; files a
    previous run wrote are set aside and used only when a patient has nothing
    else, so re-running on an output folder still works while a folder holding
    both an original and its registration uses the original.
    """
    fresh: dict = {}
    previous: dict = {}
    for directory, _, file_names in os.walk(root):
        relative = os.path.relpath(directory, root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith(".") or not accept(file_name):
                continue
            # A file that says it is a SEGMENTATION is not a subject. Only the
            # extension was tested, so a folder holding a scan beside the masks
            # of a previous run -- `<scan>_SegOut/`, which is what AMASSS and
            # this tool both write -- offered four subjects where there is one,
            # and each mask was then paired, segmented again and registered.
            # `discover_masks` has always required this token; its absence here
            # was the asymmetry.
            # The RAW stem, not `patient_stem`: that one strips exactly these
            # markers ("_Seg", "_lm_Pred"), so asking it would always answer no.
            if has_token(split_scan_extension(file_name)[0], catalogs.MASK_TOKENS):
                continue
            key = os.path.join(prefix, patient_stem(file_name))
            target = previous if is_previous_output(file_name, suffix) else fresh
            target.setdefault(key, os.path.join(directory, file_name))

    for key, path in previous.items():
        fresh.setdefault(key, path)
    return fresh


class Pairing:
    """What `pair()` found: the matched subjects, and what was left over.

    The leftovers are carried rather than dropped because "34 of your 40
    patients were registered" and "the other 6 are in the T2 folder under names
    nothing in T1 matched" are the same sentence, and only the second half is
    actionable. The original logged `Error: There is no patient to process.
    Check the files names.` and returned nothing.
    """

    def __init__(self, matched: dict, t1_only: list, t2_only: list):
        self.matched = matched
        self.t1_only = t1_only
        self.t2_only = t2_only

    def __len__(self) -> int:
        return len(self.matched)

    def unmatched_report(self) -> dict:
        return {"t1_without_t2": self.t1_only, "t2_without_t1": self.t2_only}


def timepoints_in(root: str, accept=is_scan_file) -> set:
    """The timepoint tokens the scans under `root` carry, e.g. {"t1", "t2"}.

    Read off the file NAMES and the directories above them, both: the shipped
    cohorts say it in the folder (`CBCT_FullyAuto/T1/...`) and a clinician's
    own export usually says it in the file.
    """
    seen = set()
    for directory, _, file_names in os.walk(root):
        parts = os.path.relpath(directory, root).split(os.sep)
        for file_name in file_names:
            if file_name.startswith(".") or not accept(file_name):
                continue
            if has_token(split_scan_extension(file_name)[0], catalogs.MASK_TOKENS):
                continue
            stem = split_scan_extension(file_name)[0]
            for part in (*parts, stem):
                seen |= {token for token in catalogs.TIMEPOINT_TOKENS
                         if has_token(part, (token,))}
    return seen


def timepoint_root(root: str, label: str, accept=is_scan_file) -> str:
    """`<root>/<label>` when a cohort keeps that timepoint in its own subfolder.

    Every cohort this family ships is `<name>/{T1,T2}/`, and a hosted-file
    picker offers the COHORT -- there is nothing else to offer. Pointing T1 at
    it paired timepoint against timepoint, so refusing it was right and left
    the shipped data unusable from the panel: no choice in the list was valid.

    Descending is not a guess. The subfolder is named after the timepoint being
    asked for, and it is only looked for when the folder given holds more than
    one timepoint -- an ordinary single-timepoint folder is never touched.
    """
    if not os.path.isdir(root):
        return root
    for entry in sorted(os.listdir(root)):
        if entry.lower() != label.lower():
            continue
        candidate = os.path.join(root, entry)
        if os.path.isdir(candidate) and discover(candidate, "", accept):
            return candidate
    return root


def pair(t1_root: str, t2_root: str, suffix: str, accept=is_scan_file) -> Pairing:
    """Match the subjects of two timepoint folders by name.

    Each folder is ONE timepoint. A folder holding both -- which is the shape
    every shipped test cohort has, `CBCT_FullyAuto/{T1,T2}/` -- is refused
    rather than paired: handed one as T1 and another as T2, this matched
    `T1/C_0001` with `T1/C_0001` and `T2/C_0001` with `T2/C_0001`, registering
    a baseline onto a baseline and reporting a success. Nothing downstream can
    notice that; the names line up perfectly.
    """
    # The DESCENT is the caller's, not this function's: every step of a run
    # has to agree on which folder a timepoint is, and doing it here left the
    # segmentation looking at the whole cohort while the pairing looked at one
    # timepoint -- masks keyed `T1/C_0001`, subjects keyed `C_0001`, and a run
    # that segmented twice and registered nothing. See `timepoint_root`.
    for label, root in (("T1", t1_root), ("T2", t2_root)):
        found = timepoints_in(root, accept)
        if len(found) > 1:
            raise ToolInputError(
                f"The {label} folder holds scans from more than one timepoint "
                f"({', '.join(sorted(t.upper() for t in found))}). Each side of a "
                f"registration is ONE timepoint: point {label} at that folder's "
                f"{label} subfolder rather than at the whole cohort, or the two "
                f"will be paired timepoint against timepoint and the run will "
                f"look like it worked."
            )

    t1 = discover(t1_root, suffix, accept)
    t2 = discover(t2_root, suffix, accept)

    matched = {
        key: {"t1": t1[key], "t2": t2[key]} for key in sorted(set(t1) & set(t2))
    }
    return Pairing(
        matched=matched,
        t1_only=sorted(set(t1) - set(t2)),
        t2_only=sorted(set(t2) - set(t1)),
    )


def discover_masks(root: str, region: str) -> dict:
    """{patient key: mask path} for one registration region.

    A file counts as a mask for `region` when its stem carries BOTH a
    segmentation token (mask/seg/pred) and one of that region's own tokens --
    or when it is the only candidate in a folder that holds nothing but masks,
    which is what AMASSS's own output looks like (`P1_seg_CBMASK.nii.gz`).
    """
    found: dict = {}
    wanted = catalogs.REGION_TOKENS[region]
    anatomy = ANATOMY_TOKENS | set(catalogs.MASK_TOKENS)

    for directory, _, file_names in os.walk(root):
        relative = os.path.relpath(directory, root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith(".") or not is_scan_file(file_name):
                continue
            stem, _ = split_scan_extension(file_name)
            if not has_token(stem, catalogs.MASK_TOKENS):
                continue
            if not has_token(stem, wanted):
                continue
            key = os.path.join(prefix, patient_stem(file_name, also_drop=anatomy))
            found.setdefault(key, os.path.join(directory, file_name))
    return found

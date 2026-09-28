"""A scan oriented into a frame, and the mask made from it, are one patient.

The defect these cover was paid for in full before it was found: a VFACE run
resampled a cohort, oriented it into two frames, segmented both, registered
three regions -- and then failed with `no Cranial base mask for this subject`,
about the very mask it had just made.

One naming convention, read two ways. `ASO` appends the frame it oriented into,
because its caller names the suffix, so VFACE's scans come back as
`C_0002_T1_CB_Or.nii.gz`. `patient_stem` truncated at `_Or` and therefore kept
that `CB`: the scan was subject `C_0002_CB`. `AMASSS` then writes
`C_0002_T1_CB_Or_seg_CBMASK.nii.gz`, and `discover_masks` drops every anatomy
token from a mask's name -- the frame with them -- so the mask was subject
`C_0002`. `AREG_CBCT.pipeline.find_masks` compared the two and matched nothing.

**The invariant, whichever side is made to move: a scan and the mask made from
that same scan key to the same subject.** This branch makes the SCAN drop the
frame, which is what the three other identity rules in this family already do:

    AutoMatrix.PATIENT_TOKENS_TO_DROP     drops cb / mand / max
    sadt_vface.landmarks.DECORATION_TOKENS   stops the identifier at the first
    pairing.discover_masks                drops them from a mask's name

The other candidate, `fix/pairing-mask-keeps-its-frame`, moves the mask side
instead; the first test below passes on both, and only
`test_the_kept_key_drops_the_frame` distinguishes them.
"""

import os

from sadt_areg_common import catalogs, pairing


def _touch(path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    open(str(path), "w").close()


def _amasss_output(root, scan_stem, structure):
    """What AMASSS writes for one scan: its own folder, one mask per structure."""
    _touch(os.path.join(str(root), f"{scan_stem}_seg_SegOut",
                        f"{scan_stem}_seg_{structure}.nii.gz"))


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------

def test_an_oriented_scan_and_its_own_mask_key_to_one_subject(tmp_path):
    """The whole defect, in two calls and one assertion.

    The leaf, not the full key: AMASSS writes one folder per scan, so a mask's
    key is always `<scan>_seg_SegOut/<subject>` while the scan's is just
    `<subject>`. That difference is expected and `find_masks` handles it; the
    subject on the end of it is what has to agree.
    """
    scans = tmp_path / "oriented"
    _touch(scans / "C_0002_T1_CB_Or.nii.gz")
    _amasss_output(scans, "C_0002_T1_CB_Or", "CBMASK")

    scan_key = next(iter(pairing.discover(str(scans), "Reg")))
    mask_key = next(iter(pairing.discover_masks(str(scans), "CB")))

    assert os.path.basename(mask_key) == scan_key


def test_the_kept_key_drops_the_frame(tmp_path):
    """This branch's choice, stated once so a reader can see which one it is.

    A subject is a subject in both frames. The consequence to know about is at
    the bottom of this file: two views of one subject in ONE folder become one
    key, so a caller who mixes the frames loses one. VFACE never does --
    `dispatch._split_by_frame` gives each frame its own folder, precisely
    because every step after reads one frame at a time.
    """
    scans = tmp_path / "oriented"
    _touch(scans / "C_0002_T1_CB_Or.nii.gz")
    _amasss_output(scans, "C_0002_T1_CB_Or", "CBMASK")

    assert pairing.patient_stem("C_0002_T1_CB_Or.nii.gz") == "C_0002"
    assert list(pairing.discover(str(scans), "Reg")) == ["C_0002"]
    assert [os.path.basename(key)
            for key in pairing.discover_masks(str(scans), "CB")] == ["C_0002"]


def test_the_mandible_is_masked_in_the_cranial_base_frame(tmp_path):
    """`C_0001_T1_CB_Or_seg_MANDMASK.nii.gz` -- the shipped reference's own name.

    A mandible is measured by how it sits relative to the skull, so VFACE works
    it in the CRANIAL BASE frame. The mask therefore names one anatomy in its
    frame and another in its structure, and both have to be read correctly at
    once: the frame goes, the structure goes, the subject stays.
    """
    scans = tmp_path / "oriented"
    _touch(scans / "C_0001_T1_CB_Or.nii.gz")
    _amasss_output(scans, "C_0001_T1_CB_Or", "MANDMASK")

    assert [os.path.basename(key)
            for key in pairing.discover_masks(str(scans), "MAND")] == ["C_0001"]


# ---------------------------------------------------------------------------
# What must not move with it
# ---------------------------------------------------------------------------

def test_a_structure_token_is_still_not_part_of_the_identity(tmp_path):
    """`P1_T1_MAND_seg.nii.gz` names what the mask COVERS, and still keys to P1."""
    _touch(tmp_path / "P1_T1.nii.gz")
    _touch(tmp_path / "P1_T1_MAND_seg.nii.gz")

    assert list(pairing.discover(str(tmp_path), "Reg")) == ["P1"]
    assert list(pairing.discover_masks(str(tmp_path), "MAND")) == ["P1"]


def test_aregs_own_oriented_mode_is_untouched(tmp_path):
    """`C_0001_T1_Or.nii.gz` names no frame, so there is nothing to drop.

    This is the name AREG's `Oriented + Fully-Automated` mode works on, and its
    runs were validated against the legacy module the same day this landed. The
    key it produced then is the key it produces now.
    """
    assert pairing.patient_stem("C_0001_T1_Or.nii.gz") == "C_0001"
    assert pairing.patient_stem("P1_Or.nii.gz") == "P1"
    assert pairing.patient_stem("P1_OR.nii.gz") == "P1"
    assert pairing.patient_stem("C_0001_T1_Or_CBMASK-Seg_Pred.nii.gz") == "C_0001"


def test_an_identifier_that_merely_holds_a_region_word_is_left_alone():
    """Only the ONE token immediately before the suffix, and only a region word.

    `MAX_01` is somebody's identifier: the token before `_Or` is `01`, so
    nothing is dropped and the subject keeps its name. `P_Orion` is the older
    trap, and the boundary rule that fixed it still holds.
    """
    assert pairing.patient_stem("MAX_01_T1_Or.nii.gz") == "MAX_01"
    assert pairing.patient_stem("P_Orion_T1.nii.gz") == "P_Orion"
    assert pairing.patient_stem("SMITH_ORTHO_T1.nii.gz") == "SMITH_ORTHO"
    # Two region words running: only the last of them is the frame.
    assert pairing.patient_stem("MAX_01_T1_CB_Or.nii.gz") == "MAX_01"


def test_a_file_that_is_only_a_frame_keeps_something(tmp_path):
    """`CB_Or.nii.gz` names no subject, and must not key to the empty string.

    An empty key would collapse every such file into one patient -- the very
    failure the module's docstring opens on, in its worst form.
    """
    assert pairing.patient_stem("CB_Or.nii.gz") == "CB"


def test_every_orientation_suffix_is_covered():
    """The table, not a hand-written list of three cases."""
    for suffix in catalogs.ORIENTATION_SUFFIXES:
        assert pairing.patient_stem(f"C_0002_T1_CB{suffix}.nii.gz") == "C_0002", suffix


# ---------------------------------------------------------------------------
# The consequence this branch accepts, written down
# ---------------------------------------------------------------------------

def test_two_frames_of_one_subject_in_one_folder_become_one_key(tmp_path):
    """Documented, not desired -- and the reason `_split_by_frame` exists.

    Handed both views of a subject in a single folder, `discover` keeps the
    first in sorted order and drops the other, exactly as it does for any two
    files that key alike. The candidate branch
    `fix/pairing-mask-keeps-its-frame` does not have this property; it is the
    price of agreeing with the three other identity rules.
    """
    _touch(tmp_path / "C_0002_T1_CB_Or.nii.gz")
    _touch(tmp_path / "C_0002_T1_MAX_Or.nii.gz")

    found = pairing.discover(str(tmp_path), "Reg")
    assert list(found) == ["C_0002"]
    assert found["C_0002"].endswith("C_0002_T1_CB_Or.nii.gz")

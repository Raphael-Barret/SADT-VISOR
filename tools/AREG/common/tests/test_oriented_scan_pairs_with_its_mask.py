"""A scan oriented into a frame, and the mask made from it, are one patient.

The defect these cover was paid for in full before it was found: a VFACE run
resampled a cohort, oriented it into two frames, segmented both, registered
three regions -- and then failed with `no Cranial base mask for this subject`,
about the very mask it had just made.

One naming convention, read two ways. `ASO` appends the frame it oriented into,
because its caller names the suffix, so VFACE's scans come back as
`C_0002_T1_CB_Or.nii.gz`. `patient_stem` truncates at `_Or` and therefore keeps
that `CB`: the scan is subject `C_0002_CB`. `AMASSS` then writes
`C_0002_T1_CB_Or_seg_CBMASK.nii.gz`, and `discover_masks` drops every anatomy
token from a mask's name -- the frame with them -- so the mask is subject
`C_0002`. Nothing downstream can recover: `AREG_CBCT.pipeline.find_masks` falls
back to the leaf of a key, compares `C_0002_CB` with `C_0002`, and matches
nothing.

**The invariant, whichever side is made to move: a scan and the mask made from
that same scan key to the same subject.** This branch makes the mask keep the
frame. The other candidate, `fix/pairing-frame-is-not-identity`, makes the scan
drop it; the first test below passes on both, and only
`test_the_kept_key_carries_the_frame` distinguishes them.
"""

import os

from sadt_areg_common import pairing


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


def test_the_kept_key_carries_the_frame(tmp_path):
    """This branch's choice, stated once so a reader can see which one it is.

    The scan side is left exactly as it was -- `patient_stem` is not touched --
    and the mask side is brought up to it. That is what makes this the
    conservative candidate: `AREG_CBCT`'s scan keys, whose runs were validated
    against the legacy module, do not move at all.
    """
    scans = tmp_path / "oriented"
    _touch(scans / "C_0002_T1_CB_Or.nii.gz")
    _amasss_output(scans, "C_0002_T1_CB_Or", "CBMASK")

    assert list(pairing.discover(str(scans), "Reg")) == ["C_0002_CB"]
    assert pairing.patient_stem("C_0002_T1_CB_Or.nii.gz") == "C_0002_CB"
    assert [os.path.basename(key)
            for key in pairing.discover_masks(str(scans), "CB")] == ["C_0002_CB"]


# ---------------------------------------------------------------------------
# What must not move with it
# ---------------------------------------------------------------------------

def test_a_structure_token_is_still_not_part_of_the_identity(tmp_path):
    """`P1_T1_MAND_seg.nii.gz` names what the mask COVERS, not a frame.

    This is the case `also_drop` was introduced for, and the one a fix that
    simply stopped dropping anatomy would break: the mask would key to
    `P1_MAND` and pair with no scan at all.
    """
    _touch(tmp_path / "P1_T1.nii.gz")
    _touch(tmp_path / "P1_T1_MAND_seg.nii.gz")

    assert list(pairing.discover(str(tmp_path), "Reg")) == ["P1"]
    assert list(pairing.discover_masks(str(tmp_path), "MAND")) == ["P1"]


def test_a_cohort_that_was_never_oriented_is_untouched(tmp_path):
    """No `_Or` in the name, nothing to protect, same keys as before."""
    scans = tmp_path / "T1"
    _touch(scans / "C_0001_T1.nii.gz")
    _amasss_output(scans, "C_0001_T1", "CBMASK")

    assert list(pairing.discover(str(scans), "Reg")) == ["C_0001"]
    assert [os.path.basename(key)
            for key in pairing.discover_masks(str(scans), "CB")] == ["C_0001"]


def test_the_two_frames_of_one_subject_do_not_collapse(tmp_path):
    """A frame is kept, so the two views of `C_0002` stay two keys.

    VFACE never has both in one folder -- `_split_by_frame` puts each in its own
    -- but the masks of a cohort a caller sends could be laid out either way,
    and one of the two silently winning is the failure mode `discover`'s
    `setdefault` would produce.
    """
    for frame, structure in (("CB", "CBMASK"), ("MAX", "MAXMASK")):
        _amasss_output(tmp_path, f"C_0002_T1_{frame}_Or", structure)

    keys = {os.path.basename(key) for key in pairing.discover_masks(str(tmp_path), "CB")}
    keys |= {os.path.basename(key) for key in pairing.discover_masks(str(tmp_path), "MAX")}
    assert keys == {"C_0002_CB", "C_0002_MAX"}


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def test_only_a_token_an_orientation_follows_counts_as_a_frame():
    """The position is the whole rule: `CB` before `_Or`, and nowhere else."""
    assert pairing.frame_tokens("C_0002_T1_CB_Or_seg_CBMASK") == {"cb"}
    assert pairing.frame_tokens("C_0002_T1_MAX_Or_seg_MAXMASK") == {"max"}
    # A structure, not a frame.
    assert pairing.frame_tokens("P1_T1_MAND_seg") == frozenset()
    # An orientation that named no frame -- AREG's own oriented mode.
    assert pairing.frame_tokens("C_0001_T1_Or") == frozenset()
    # A subject whose name merely ends in something orientation-shaped.
    assert pairing.frame_tokens("P_Orion_T1") == frozenset()


def test_a_subject_named_after_a_region_is_not_given_a_frame():
    """`MAX_01` is somebody's identifier, and the token before `_Or` is `01`.

    The rule reads one position, so a region word anywhere else in a name is
    not promoted to a frame by it.
    """
    assert pairing.frame_tokens("MAX_01_T1_Or") == frozenset()
    assert pairing.frame_tokens("MAX_01_T1_CB_Or") == {"cb"}

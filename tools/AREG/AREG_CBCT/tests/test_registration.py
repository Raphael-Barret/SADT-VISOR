"""What a run produces: the tree, the names, the report, and what it does not
produce.

elastix is real here. The one thing consistently NOT asserted is the numeric
quality of the registration -- `test_run.py::TestElastix` owns that; these are
about the envelope around it, which is where the port's differences from the
Slicer module live.
"""

import json
import os

import numpy as np
import pytest
import SimpleITK as sitk

from conftest import cohort, displaced, full_mask, phantom, tree_of, write
from sadt_areg_cbct import dispatch
from sadt_areg_common import catalogs, pairing
from sadt_areg_common.errors import ToolInputError


def semi(root, regions=("Cranial base",), **overrides):
    arguments = {
        "t1_path": os.path.join(str(root), "T1"),
        "t2_path": os.path.join(str(root), "T2"),
        "t1_masks_path": os.path.join(str(root), "masks"),
        "automation": catalogs.AUTOMATION_SEMI,
        "regions": list(regions),
        "output_dir": os.path.join(str(root), "out"),
    }
    arguments.update(overrides)
    return dispatch.register(**arguments)


# ---------------------------------------------------------------------------
# The output tree
# ---------------------------------------------------------------------------

def test_a_run_writes_a_registered_scan_a_transform_and_a_report(semi_run):
    run, _root = semi_run
    assert run.succeeded == ["P1"]
    assert tree_of(run.output_dir) == sorted([
        "AREG_report.json",
        os.path.join("CB", "P1_CB_Reg.nii.gz"),
        os.path.join("CB", "P1_CB_Reg_transform.tfm"),
    ])


def test_each_region_is_a_registration_of_its_own(tmp_path):
    """Registering on the cranial base and on the mandible are two different
    clinical questions, not two settings of one -- so each gets its own output
    folder and its own report entry."""
    cohort(tmp_path, regions=("CB", "MAND"), size=32)
    run = semi(tmp_path, regions=("Cranial base", "Mandible"))

    assert tree_of(run.output_dir) == sorted([
        "AREG_report.json",
        os.path.join("CB", "P1_CB_Reg.nii.gz"),
        os.path.join("CB", "P1_CB_Reg_transform.tfm"),
        os.path.join("MAND", "P1_MAND_Reg.nii.gz"),
        os.path.join("MAND", "P1_MAND_Reg_transform.tfm"),
    ])
    assert set(run.patients["P1"]["regions"]) == {"CB", "MAND"}


def test_a_batch_keeps_the_tree_its_scans_arrived_in(tmp_path):
    """Two subjects called `P1` at two sites are two patients. The original
    keyed on the base name, so one silently overwrote the other -- in the
    working dictionary and again in the flat output folder."""
    for site in ("siteA", "siteB"):
        fixed = phantom(size=32, seed=abs(hash(site)) % 100)
        moving, _truth = displaced(fixed)
        write(fixed, str(tmp_path / "T1" / site / "P1_T1_scan.nii.gz"))
        write(moving, str(tmp_path / "T2" / site / "P1_T2_scan.nii.gz"))
        write(full_mask(fixed), str(tmp_path / "masks" / site / "P1_T1_CB_seg.nii.gz"))

    run = semi(tmp_path)
    assert sorted(run.succeeded) == [os.path.join("siteA", "P1"), os.path.join("siteB", "P1")]
    assert tree_of(run.output_dir) == sorted([
        "AREG_report.json",
        os.path.join("CB", "siteA", "P1_CB_Reg.nii.gz"),
        os.path.join("CB", "siteA", "P1_CB_Reg_transform.tfm"),
        os.path.join("CB", "siteB", "P1_CB_Reg.nii.gz"),
        os.path.join("CB", "siteB", "P1_CB_Reg_transform.tfm"),
    ])


def test_the_suffix_reaches_both_names_it_writes(tmp_path):
    cohort(tmp_path, size=32)
    run = semi(tmp_path, output_suffix="Study7")
    assert tree_of(run.output_dir) == sorted([
        "AREG_report.json",
        os.path.join("CB", "P1_CB_Study7.nii.gz"),
        os.path.join("CB", "P1_CB_Study7_transform.tfm"),
    ])


@pytest.mark.parametrize(
    "given,expected", [(".nii", ".nii.gz"), (".nrrd", ".nrrd"), (".nii.gz", ".nii.gz")]
)
def test_the_output_carries_a_spelling_itk_can_actually_write(tmp_path, given, expected):
    """NIfTI and GIPL take an external `.gz`; NRRD compresses inside the file
    and ITK has no `.nrrd.gz` writer at all, so that spelling maps back down."""
    cohort(tmp_path, size=32, extension=given)
    run = semi(tmp_path)
    assert os.path.join("CB", f"P1_CB_Reg{expected}") in tree_of(run.output_dir)


def test_the_output_directory_is_created_when_it_does_not_exist(tmp_path):
    cohort(tmp_path, size=32)
    run = semi(tmp_path, output_dir=str(tmp_path / "deep" / "nested" / "out"))
    assert os.path.isfile(os.path.join(run.output_dir, "AREG_report.json"))


def test_nothing_is_written_outside_the_output_directory(tmp_path, monkeypatch):
    """The Slicer module wrote a `<t2_folder>_Center` directory next to the
    caller's own data, a log file the client polled, and `*Error.txt` files
    into the output folder. None of that survives."""
    monkeypatch.chdir(tmp_path)
    cohort(tmp_path, size=32)
    before = {name: tree_of(tmp_path / name) for name in ("T1", "T2", "masks")}

    semi(tmp_path)

    assert {name: tree_of(tmp_path / name) for name in before} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["T1", "T2", "masks", "out"]


def test_the_working_directory_does_not_survive_the_run(semi_run):
    """A surviving `.areg_work/` means a run crashed -- and it holds the
    extracted inputs, the converted DICOM and whatever the tools it drove
    wrote, all of which would be shipped to the client."""
    run, _root = semi_run
    assert not os.path.exists(os.path.join(run.output_dir, dispatch.WORK_DIRNAME))


def test_the_registered_volume_keeps_the_t2s_own_grid(semi_run):
    """Resampled on the MOVING image's grid, as the original did: the result
    keeps the T2's resolution and field of view, in the T1's frame. Resampling
    onto the T1 grid instead would crop the T2 to the T1's field of view."""
    run, root = semi_run
    moving = sitk.ReadImage(os.path.join(root, "T2", "P1_T2_scan.nii.gz"))
    registered = sitk.ReadImage(os.path.join(run.output_dir, "CB", "P1_CB_Reg.nii.gz"))

    assert registered.GetSize() == moving.GetSize()
    assert registered.GetSpacing() == pytest.approx(moving.GetSpacing())
    assert registered.GetPixelID() == sitk.sitkInt16


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_lands_beside_the_results_and_says_what_ran(semi_run):
    run, _root = semi_run
    with open(os.path.join(run.output_dir, dispatch.REPORT_NAME), encoding="utf-8") as handle:
        report = json.load(handle)

    assert report["modality"] == "CBCT"
    assert report["automation"] == catalogs.AUTOMATION_SEMI
    assert report["output_suffix"] == "Reg"
    assert report["regions"] == ["Cranial base"]
    assert report["summary"] == {"patients": 1, "registered": 1, "failed": 0}
    assert report["unmatched"] == {"t1_without_t2": [], "t2_without_t1": []}


def test_the_report_says_which_way_the_transform_maps(semi_run):
    """Stated rather than assumed: getting the direction backwards is silent --
    the file still loads and still transforms -- and it is the only thing a
    downstream tool needs to know to reuse it."""
    run, _root = semi_run
    entry = run.patients["P1"]["regions"]["CB"]
    assert entry["transform_maps"] == (
        "T1 space -> T2 space (what sitk.ResampleImageFilter consumes)"
    )


def test_the_report_names_the_mask_each_registration_actually_used(semi_run):
    """Which anatomy a registration was confined to is the one thing that
    decides whether the answer means anything."""
    run, _root = semi_run
    entry = run.patients["P1"]["regions"]["CB"]
    assert entry["mask"] == "P1_T1_CB_seg.nii.gz"
    assert entry["region"] == "Cranial base"


def test_the_report_lists_what_the_run_produced_relative_to_the_output(semi_run):
    """The report travels inside the archive. The job directory it was
    produced in does not."""
    run, root = semi_run
    entry = run.patients["P1"]["regions"]["CB"]
    assert entry["outputs"] == sorted([
        os.path.join("CB", "P1_CB_Reg.nii.gz"),
        os.path.join("CB", "P1_CB_Reg_transform.tfm"),
    ])
    with open(os.path.join(run.output_dir, dispatch.REPORT_NAME), encoding="utf-8") as handle:
        assert root not in handle.read()


def test_a_multi_label_mask_used_whole_says_so_in_the_report(tmp_path):
    """`segmentation_label` 0 means "all of it", which is a decision the reader
    of the result has to be able to see."""
    fixed = phantom(size=32)
    moving, _truth = displaced(fixed)
    write(fixed, str(tmp_path / "T1" / "P1_T1_scan.nii.gz"))
    write(moving, str(tmp_path / "T2" / "P1_T2_scan.nii.gz"))

    array = np.zeros(sitk.GetArrayViewFromImage(fixed).shape, np.uint8)
    array[4:20] = 1
    array[20:28] = 2
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(fixed)
    write(mask, str(tmp_path / "masks" / "P1_T1_CB_seg.nii.gz"))

    run = semi(tmp_path)
    assert "several labels" in run.patients["P1"]["regions"]["CB"]["note"]


def test_naming_a_label_registers_on_that_label_alone(tmp_path):
    fixed = phantom(size=32)
    moving, _truth = displaced(fixed)
    write(fixed, str(tmp_path / "T1" / "P1_T1_scan.nii.gz"))
    write(moving, str(tmp_path / "T2" / "P1_T2_scan.nii.gz"))

    array = np.zeros(sitk.GetArrayViewFromImage(fixed).shape, np.uint8)
    array[4:24] = 1
    array[24:30] = 2
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(fixed)
    write(mask, str(tmp_path / "masks" / "P1_T1_CB_seg.nii.gz"))

    run = semi(tmp_path, segmentation_label=2)
    entry = run.patients["P1"]["regions"]["CB"]
    assert entry["status"] == "ok"
    assert "note" not in entry
    assert run.report["segmentation_label"] == 2


def test_a_label_the_mask_does_not_hold_fails_that_patient_only(tmp_path):
    """It used to fall through to using the WHOLE mask, so asking for label 4
    of a two-label mask registered on everything and reported success."""
    cohort(tmp_path, subjects=("P1", "P2"), size=32)
    run = semi(tmp_path, segmentation_label=4)

    assert run.succeeded == []
    for key in ("P1", "P2"):
        assert "no label 4" in run.patients[key]["regions"]["CB"]["reason"]


# ---------------------------------------------------------------------------
# Failure, per patient
# ---------------------------------------------------------------------------

def test_a_subject_whose_mask_is_a_different_sampling_is_reported(tmp_path):
    """The original forced `mask.SetOrigin(image.GetOrigin())` unconditionally,
    so a mask that is genuinely a different sampling of the patient was applied
    several millimetres off, in silence."""
    cohort(tmp_path, subjects=("P1", "P2"), size=32)
    write(full_mask(phantom(size=24)), str(tmp_path / "masks" / "P2_T1_CB_seg.nii.gz"))

    run = semi(tmp_path)
    assert run.succeeded == ["P1"]
    assert "not the same sampling" in run.patients["P2"]["regions"]["CB"]["reason"]


def test_a_scan_that_cannot_be_read_is_reported_not_raised(tmp_path):
    """One corrupt file in a cohort must not abort the thirty-nine that
    read."""
    cohort(tmp_path, subjects=("P1", "P2"), size=32)
    with open(str(tmp_path / "T2" / "P2_T2_scan.nii.gz"), "wb") as handle:
        handle.write(b"not a volume")

    run = semi(tmp_path)
    assert run.succeeded == ["P1"]
    assert run.patients["P2"]["regions"]["CB"]["status"] == "failed"


def test_a_patient_is_ok_when_any_of_its_regions_registered(tmp_path):
    """Two regions of one patient are two registrations, and one can fail on
    its own without the patient's result being worthless."""
    cohort(tmp_path, regions=("CB",), size=32)
    run = semi(tmp_path, regions=("Cranial base", "Mandible"))

    assert run.succeeded == ["P1"]
    assert run.patients["P1"]["regions"]["CB"]["status"] == "ok"
    assert run.patients["P1"]["regions"]["MAND"]["status"] == "failed"


def test_a_subject_with_no_mask_says_how_to_name_one(tmp_path):
    """A mask is matched to its scan by name, and the message has to say both
    halves of the rule -- otherwise the fix is guesswork."""
    cohort(tmp_path, size=32)
    os.remove(str(tmp_path / "masks" / "P1_T1_CB_seg.nii.gz"))

    run = semi(tmp_path)
    reason = run.patients["P1"]["regions"]["CB"]["reason"]
    assert "no Cranial base mask" in reason
    assert "mask/seg/pred" in reason
    assert "P1_T1_CB_seg.nii.gz" in reason


def test_a_missing_mask_in_an_automated_mode_points_at_the_segmentation_step(tmp_path):
    """The same hole, a different fix: in Semi-Automated the caller names the
    mask, in the automated modes the segmentation was supposed to make one."""
    reason = dispatch._no_mask_reason(catalogs.AUTOMATION_FULLY, "CB")
    assert "segmentation step" in reason
    assert "AMASSS" in reason


# ---------------------------------------------------------------------------
# Pairing, at the level the run depends on
# ---------------------------------------------------------------------------

def test_a_second_run_does_not_take_the_first_ones_output_as_input(tmp_path):
    """`P1_CB_Reg.nii.gz` sorts before `P1_T2_scan.nii.gz`, so without the
    previous-output rule a re-run would register an already-registered scan."""
    cohort(tmp_path, size=16)
    write(phantom(size=16), str(tmp_path / "T2" / "P1_CB_Reg.nii.gz"))

    found = pairing.discover(str(tmp_path / "T2"), "Reg")
    assert os.path.basename(found["P1"]) == "P1_T2_scan.nii.gz"


def test_a_previous_output_is_still_usable_when_it_is_all_there_is(tmp_path):
    """Set aside, not excluded: re-running on an output folder has to work.

    It keys under its own full stem -- `_Reg` is the caller's suffix, not one
    of the fixed `PATIENT_SUFFIXES` -- so both timepoints of a re-run carry it
    and still pair with each other.
    """
    write(phantom(size=16), str(tmp_path / "T2" / "P1_CB_Reg.nii.gz"))
    found = pairing.discover(str(tmp_path / "T2"), "Reg")
    assert [os.path.basename(path) for path in found.values()] == ["P1_CB_Reg.nii.gz"]


def test_a_subject_in_only_one_timepoint_is_named_in_the_report(tmp_path):
    """"34 of your 40 patients were registered" and "the other 6 are in the T2
    folder under names nothing in T1 matched" are the same sentence, and only
    the second half is actionable."""
    cohort(tmp_path, subjects=("P1",), size=32)
    write(phantom(size=32), str(tmp_path / "T2" / "P9_T2_scan.nii.gz"))

    run = semi(tmp_path)
    assert run.report["unmatched"] == {"t1_without_t2": [], "t2_without_t1": ["P9"]}


def test_no_pair_at_all_names_the_rule_and_the_counts(tmp_path):
    write(phantom(size=16), str(tmp_path / "T1" / "alpha_T1.nii.gz"))
    write(phantom(size=16), str(tmp_path / "T2" / "beta_T2.nii.gz"))

    with pytest.raises(ToolInputError) as raised:
        semi(tmp_path, t1_masks_path=str(tmp_path / "T1"))
    message = str(raised.value)
    assert "paired by name" in message
    assert "1 T1-only and 1 T2-only" in message


def test_a_single_file_input_becomes_a_folder_of_its_own(tmp_path):
    """`main.py` streams every upload of a request into ONE work directory, so
    treating a file's parent as an input root would make the T2 folder part of
    the T1 one."""
    fixed = phantom(size=32)
    moving, _truth = displaced(fixed)
    both = tmp_path / "uploads"
    write(fixed, str(both / "P1_T1_scan.nii.gz"))
    write(moving, str(both / "P1_T2_scan.nii.gz"))
    write(full_mask(fixed), str(tmp_path / "masks" / "P1_T1_CB_seg.nii.gz"))

    run = semi(
        tmp_path,
        t1_path=str(both / "P1_T1_scan.nii.gz"),
        t2_path=str(both / "P1_T2_scan.nii.gz"),
    )
    assert run.succeeded == ["P1"]


# ---------------------------------------------------------------------------
# The mode, read off the request
# ---------------------------------------------------------------------------
# `automation` used to be a dropdown a clinician had to set, and setting it
# wrong was silent: Fully-Automated over a folder of masks segmented again,
# Semi-Automated over scans with none failed patient by patient. Each of the
# three modes is identified by an INPUT, so the question never needed asking.
# ASO closed the same hole first, for landmarks.


def test_masks_make_it_semi_automated():
    mode, source = dispatch.derive_automation(
        catalogs.AUTOMATION_AUTO, "/somewhere/masks", None
    )

    assert mode == catalogs.AUTOMATION_SEMI
    assert source == "from the data"


def test_no_masks_makes_it_fully_automated():
    mode, source = dispatch.derive_automation(catalogs.AUTOMATION_AUTO, None)

    assert mode == catalogs.AUTOMATION_FULLY
    assert source == "from the data"


@pytest.mark.parametrize("frame", [
    catalogs.ORIENTATION_FRANKFURT, catalogs.ORIENTATION_OCCLUSAL,
])
def test_naming_a_frame_is_what_picks_the_oriented_variant(frame):
    """The one thing here no folder can answer, so it is asked by NAME.

    Both frames select the oriented mode: they differ in WHERE the scans end up,
    not in whether they are oriented."""
    mode, source = dispatch.derive_automation(catalogs.AUTOMATION_AUTO, None, frame)

    assert mode == catalogs.AUTOMATION_ORIENTED
    assert source == "from the data"


def test_leaving_the_scans_as_they_came_is_fully_automated():
    assert dispatch.derive_automation(
        catalogs.AUTOMATION_AUTO, None, catalogs.ORIENTATION_NONE
    )[0] == catalogs.AUTOMATION_FULLY
    # And the same for a caller that sends nothing at all, `main` being callable
    # directly, where the signature's default does not apply.
    assert dispatch.derive_automation(catalogs.AUTOMATION_AUTO, None, None)[0] == (
        catalogs.AUTOMATION_FULLY
    )


def test_masks_win_over_asking_for_an_orientation():
    """With masks in hand there is nothing to segment and nothing to orient
    for, so the frame is not read."""
    mode, _source = dispatch.derive_automation(
        catalogs.AUTOMATION_AUTO, "/somewhere/masks", catalogs.ORIENTATION_FRANKFURT
    )

    assert mode == catalogs.AUTOMATION_SEMI


def test_each_frame_resolves_to_its_own_bundle(tmp_path):
    """A reference defines its frame through what it CARRIES, so naming the
    frame is naming the bundle -- and the two carry disjoint landmark sets, which
    is why one cannot stand in for the other."""
    models = tmp_path / "AREG" / "models"
    for bundle in catalogs.ORIENTATION_BUNDLES.values():
        (models / bundle).mkdir(parents=True)

    for frame, bundle in catalogs.ORIENTATION_BUNDLES.items():
        assert dispatch._own_reference(str(tmp_path), frame) == str(models / bundle)

    # The frame that is no frame resolves to nothing, rather than to the first
    # bundle it finds.
    assert dispatch._own_reference(str(tmp_path), catalogs.ORIENTATION_NONE) == ""


def test_the_orientation_reference_is_never_a_mode_signal():
    """Measured through the server on 2026-09-29, after 113 unit tests had
    passed on the opposite assumption.

    `reference` carries `server_selectable = "model"`, so the server fills it
    from `DATA/AREG/models/` whenever the client leaves it empty -- it is never
    absent. Reading it as "the caller asked for an orientation" refused every
    run that sent masks, for a reference nobody had named.

    Asserted on the SIGNATURE, because that is what makes the mistake
    impossible to make again: there is nowhere to pass it.
    """
    import inspect

    assert list(inspect.signature(dispatch.derive_automation).parameters) == [
        "automation", "t1_masks", "orientation",
    ]


@pytest.mark.parametrize("named", [
    catalogs.AUTOMATION_SEMI, catalogs.AUTOMATION_FULLY, catalogs.AUTOMATION_ORIENTED,
])
def test_a_named_mode_overrides_the_data(named):
    """The one thing naming a mode still does, and a legitimate thing to want:
    "Fully-Automated" beside a folder of masks means "segment anyway, I know
    they are there"."""
    mode, source = dispatch.derive_automation(named, "/somewhere/masks")

    assert mode == named
    assert source == "requested"


def test_the_report_says_whether_anybody_chose_the_mode(tmp_path):
    """"Semi-Automated" in a report does not say whether it was asked for."""
    cohort(tmp_path)
    run = semi(tmp_path, automation=catalogs.AUTOMATION_AUTO)

    assert run.report["automation"] == catalogs.AUTOMATION_SEMI
    assert run.report["automation_source"] == "from the data"


# ---------------------------------------------------------------------------
# The orientation reference the deployment answers with
# ---------------------------------------------------------------------------
# `reference` is `server_selectable = "model"`, so a request that names no
# bundle arrives holding `DATA/AREG/models/` -- the FOLDER, eight bundles deep,
# which ASO cannot orient onto. The field is hidden and the tool answers.


def test_a_deployment_without_the_bundle_says_nothing_rather_than_guessing(tmp_path):
    """Returns "" so the single refusal in `_check_cbct` keeps saying what is
    missing, instead of handing ASO a path that is not there."""
    (tmp_path / "AREG" / "models").mkdir(parents=True)

    assert dispatch._own_reference(
        str(tmp_path), catalogs.ORIENTATION_FRANKFURT
    ) == ""


def test_the_models_folder_is_not_mistaken_for_a_bundle(tmp_path):
    """The whole point: `main` has to tell "the caller named a bundle" from
    "the server filled this in with the folder holding all of them".

    Asserted through `main`, because the discrimination lives there -- and a run
    that got the folder died inside ASO, minutes in, on a path that looks
    perfectly valid.
    """
    data_root = tmp_path / "data"
    frame = catalogs.ORIENTATION_FRANKFURT
    bundle = (data_root / "AREG" / "models"
              / catalogs.ORIENTATION_BUNDLES[frame])
    bundle.mkdir(parents=True)
    cohort(tmp_path)

    with pytest.raises(Exception):  # noqa: B017 - no supervisor, so ASO is absent
        dispatch.main(
            automation=catalogs.AUTOMATION_ORIENTED,
            t1=os.path.join(str(tmp_path), "T1"),
            t2=os.path.join(str(tmp_path), "T2"),
            cbct_regions=["Cranial base"],
            cbct_reference=os.path.join(str(data_root), "AREG", "models"),
            output_dir=os.path.join(str(tmp_path), "out"),
            data_root=str(data_root),
        )
    # What matters is that the refusal is about the missing ASO tool, not about
    # a reference -- the folder was replaced by the bundle before the check.
    assert dispatch._own_reference(str(data_root), frame) == str(bundle)


def test_the_orientation_reference_is_not_put_to_the_reader():
    from sadt_areg_cbct.layout import LAYOUT

    assert LAYOUT["reference"]["hidden"] is True


def test_register_on_reads_like_the_amasss_lists():
    """Three lists of anatomy in one panel should read alike."""
    from sadt_areg_cbct.layout import LAYOUT

    assert LAYOUT["regions"]["ui"] == "chips"
    assert LAYOUT["segmentations"]["ui"] == "chips"
    # And each chip says which folder its transforms come back in.
    assert LAYOUT["regions"]["option_help"]["Cranial base"] == "CB"

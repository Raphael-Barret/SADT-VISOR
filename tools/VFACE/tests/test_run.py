"""VFACE end to end, with the six tools it drives standing in for themselves.

The chain runs for real. What `PipelineSup` stands in for is each callee's own
computation -- `ASO` writes a scan named as it would name it and the transform
beside it, `ALI_CBCT` writes the landmark files -- and everything VFACE itself
does runs on what they wrote: the resample, the padding, the derivation between
the two frames, the measurements, the signed features, the classification.

That is the difference between this and a mock of the pipeline: a wrong
argument, a folder handed to the wrong step or a landmark file that cannot be
found fails here.
"""

import json
import os

import pytest

from conftest import (
    PATIENTS, PipelineSup, cohort, tree_of,
    write_feature_template, write_measurement_lists, write_volume,
)
from sadt_vface import catalogs, dispatch, run
from sadt_vface.errors import SupervisorRequired, ToolInputError


def request(tmp_path, **overrides):
    """A full-pipeline asymmetry assessment over two patients."""
    cohort(tmp_path, patients=PATIENTS)
    arguments = {
        "t1": str(tmp_path / "t1"),
        "output_dir": str(tmp_path / "out"),
        "mode": catalogs.MODE_FULL,
        "study": catalogs.STUDY_ASYMMETRY,
        "outputs": catalogs.OUTPUT_QUANTITATIVE,
        "measurements": write_measurement_lists(tmp_path / "lists"),
        "cranial_base_reference": str(tmp_path / "models" / "cb_gold"),
        "maxilla_reference": str(tmp_path / "models" / "max_gold"),
        "mirror_reference": str(tmp_path / "models" / "mirror.tfm"),
        "segmentation_model": str(tmp_path / "models" / "amasss"),
        "landmark_model": str(tmp_path / "models" / "ali"),
    }
    arguments.update(overrides)
    return arguments


def read_report(output_dir):
    with open(os.path.join(str(output_dir), dispatch.REPORT_NAME), encoding="utf-8") as handle:
        return json.load(handle)


def read_table(path):
    import pandas as pd

    return pd.read_excel(str(path))


# ---------------------------------------------------------------------------
# The whole chain
# ---------------------------------------------------------------------------

def test_a_full_asymmetry_assessment_writes_one_table_per_region(tmp_path):
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    written = tree_of(tmp_path / "out")
    for region in ("CB", "MAND", "MAX"):
        assert os.path.join("Measurements", f"Measurements_{region}.xlsx") in written
    assert dispatch.REPORT_NAME in written


def test_every_one_of_the_six_tools_is_driven(tmp_path):
    """The widest call graph in this repository, and the run is where it is
    actually exercised: a renamed argument fails here rather than an hour into
    a job."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    called = {name for name, _params in sup.calls}
    assert called == {"ASO", "AMASSS", "AutoMatrix", "AREG_CBCT", "ALI_CBCT"}
    # Batch_Dental_Seg only on the visualisation path, which this run did not ask for.
    assert "Batch_Dental_Seg" not in called


def test_the_scans_are_oriented_once_per_frame_and_not_once_per_region(tmp_path):
    """Three regions, two frames: the mandible is worked in the cranial base's."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    assert len(sup.asked("ASO")) == len(catalogs.FRAMES) == 2
    assert {params["output_suffix"] for params in sup.asked("ASO")} == {"CB_Or", "MAX_Or"}


def test_each_region_is_registered_in_its_own_frame(tmp_path):
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    by_region = {params["regions"][0]: params for params in sup.asked("AREG_CBCT")}
    assert set(by_region) == {"Cranial base", "Mandible", "Maxilla"}
    # The mandible registers on the cranial-base-oriented scans, the maxilla on
    # the maxillary ones.
    assert by_region["Mandible"]["t1"] == by_region["Cranial base"]["t1"]
    assert by_region["Maxilla"]["t1"] != by_region["Cranial base"]["t1"]


def test_the_landmarks_are_searched_once_and_carried_into_the_other_frame(tmp_path):
    """Searching a second time costs minutes per patient and makes the same
    anatomical point land in two slightly different places."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    searches = sup.asked("ALI_CBCT")
    assert len(searches) == 1
    assert set(searches[0]["landmarks"]) == {"ROr", "RCo", "RGo", "ANS"}


def test_the_landmarks_asked_for_are_the_ones_the_lists_name(tmp_path):
    """ALI spawns one agent per landmark at about a minute each; a region's
    whole catalogue would be hours spent on points no measurement reads."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path, measurements=write_measurement_lists(
        tmp_path / "small", [("Distance between 2 points T1 T2", "Ba", "Ba")]
    )))
    assert sup.asked("ALI_CBCT")[0]["landmarks"] == ["Ba"]


def test_a_measurement_carries_both_its_magnitude_and_what_its_sign_means(tmp_path):
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    table = read_table(tmp_path / "out" / "Measurements" / "Measurements_CB.xlsx")
    assert set(table.columns) >= {
        "Patient", "Landmarks", "R-L Component", "R-L Meaning", "3D Distance"
    }
    assert sorted(table["Patient"].unique()) == ["P1", "P2"]
    assert (table["3D Distance"] > 0).all()


def test_the_mirror_is_what_a_patient_is_measured_against(tmp_path):
    """The whole point of an asymmetry assessment: there is no second scan, so
    the distance measured is from a landmark to its own reflection."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    table = read_table(tmp_path / "out" / "Measurements" / "Measurements_CB.xlsx")
    row = table[table["Landmarks"] == "ROr - ROr"].iloc[0]
    # ROr sits 28 mm off the midline, so it and its reflection are 56 apart,
    # plus the 1.5 mm the fixture nudges it by.
    assert row["R-L Component"] == pytest.approx(56.0, abs=1e-6)
    assert row["S-I Component"] == pytest.approx(1.5, abs=1e-6)


def test_nothing_is_left_in_the_working_directory(tmp_path):
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))
    assert not os.path.exists(str(tmp_path / "out" / dispatch.WORK_DIRNAME))


def test_the_report_says_what_ran(tmp_path):
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    report = read_report(tmp_path / "out")
    assert report["mode"] == catalogs.MODE_FULL
    assert report["study"] == catalogs.STUDY_ASYMMETRY
    assert report["regions"] == list(catalogs.REGIONS)
    assert report["duration_seconds"] >= 0


def test_the_run_reports_where_it_has_got_to(tmp_path):
    """A forty-patient run is an hour of other tools, and an elapsed timer says
    only that the connection is still open."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    fractions = [fraction for fraction, _message in sup.messages]
    assert fractions == sorted(fractions)
    assert fractions[0] < fractions[-1] <= 1.0
    assert all("P1" not in message and "P2" not in message
               for _fraction, message in sup.messages), "a patient must not travel in a message"


# ---------------------------------------------------------------------------
# The classification
# ---------------------------------------------------------------------------

def test_without_a_template_and_a_model_the_measurements_still_land(tmp_path):
    """They are a usable answer on their own, and a classification needs a
    template and a bundle from the same training run."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path))

    report = read_report(tmp_path / "out")
    assert "feature_template" in report["classification"]
    assert os.path.join("Measurements", "Measurements_CB.xlsx") in tree_of(tmp_path / "out")


def test_a_template_alone_gives_the_feature_table_and_no_verdict(tmp_path):
    """The two need different things. A template gives the measurements
    assembled one row per patient, which is what somebody training a model
    would ask for; the verdict on top of it needs a bundle from the same
    training run."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path, feature_template=write_feature_template(tmp_path / "template.xlsx")
    ))

    written = tree_of(tmp_path / "out")
    assert os.path.join("Measurements", dispatch.FEATURE_TABLE_NAME) in written
    assert not any(name.startswith("Classification") for name in written)

    report = read_report(tmp_path / "out")
    assert "classifier_model" in report["classification"]
    assert report["features"] == len(PATIENTS)


def test_a_model_the_table_cannot_feed_does_not_cost_the_measurements(tmp_path):
    """The failure a real run hit at 0.97, after everything else had succeeded.

    A bundle trained on features the run never measured is a real
    misconfiguration -- and until this test it was a 422 that took the
    registration, the landmark search and the measurement tables down with it.
    The server destroys a job directory when the run fails, so the clinician was
    left with the GPU minutes and nothing at all, for a verdict they could have
    read off the table by hand.

    So the refusal is recorded and the run finishes. What is asserted is the
    whole of that: the measurements and the feature table are in the archive,
    the report says plainly why there is no verdict, and there is no
    Classification folder pretending otherwise.
    """
    from test_classify import FEATURES, train

    bundle = tmp_path / "vface_models"
    trained_on = ["CB_Nothing_Nothing_RL", "CB_Absent_Absent_IS"]
    for name in ("sym_asymm.txt", "mand_asym.txt", "max_asym.txt"):
        train(bundle / name, trained_on, lambda row: row[trained_on[0]] > 0)

    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path,
        feature_template=write_feature_template(tmp_path / "template.xlsx", FEATURES),
        classifier_model=str(bundle),
    ))

    written = tree_of(tmp_path / "out")
    assert os.path.join("Measurements", "Measurements_CB.xlsx") in written
    assert os.path.join("Measurements", dispatch.FEATURE_TABLE_NAME) in written
    assert not any(name.startswith("Classification") for name in written)

    report = read_report(tmp_path / "out")
    assert "CB_Nothing_Nothing_RL" in report["classification"]
    assert "were written" in report["classification"]
    assert "classified" not in report


def test_a_template_naming_nothing_measured_does_not_cost_them_either(tmp_path):
    """The same rule one step earlier, where the table itself cannot be built.

    `build_feature_table` refuses when no patient has any measurement at all,
    and that refusal used to end the run the same way. Here the template names
    only columns from a region this run never measured, so every column is
    empty -- which the report says, column by column.
    """
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path,
        regions=[catalogs.REGION_CRANIAL_BASE],
        feature_template=write_feature_template(
            tmp_path / "template.xlsx", ["MAX_ANS_ANS_RL", "MAX_RPF_LPF_IS"]
        ),
    ))

    written = tree_of(tmp_path / "out")
    assert os.path.join("Measurements", "Measurements_CB.xlsx") in written
    report = read_report(tmp_path / "out")
    assert report.get("features") == len(PATIENTS)
    empty = report.get("features_empty", {})
    assert empty, "a column no measurement can fill has to be reported"


# ---------------------------------------------------------------------------
# Refusing at the door
# ---------------------------------------------------------------------------

def test_a_full_pipeline_without_a_supervisor_names_the_mode_that_works(tmp_path):
    with pytest.raises(SupervisorRequired) as raised:
        run(**request(tmp_path))
    assert "ASO" in str(raised.value)
    assert "File already Oriented" in str(raised.value)


def test_a_mode_that_does_not_exist_is_refused_with_the_ones_that_do(tmp_path):
    sup = PipelineSup(tmp_path)
    with pytest.raises(ToolInputError) as raised:
        run(sup=sup, **request(tmp_path, mode="Whatever"))
    assert "Full pipeline" in str(raised.value)


def test_a_longitudinal_study_with_no_follow_up_is_refused(tmp_path):
    sup = PipelineSup(tmp_path)
    with pytest.raises(ToolInputError) as raised:
        run(sup=sup, **request(tmp_path, study=catalogs.STUDY_LONGITUDINAL))
    assert "no 't2' was sent" in str(raised.value)
    assert catalogs.STUDY_ASYMMETRY in str(raised.value)


def test_an_asymmetry_assessment_with_no_mirror_transform_is_refused(tmp_path):
    sup = PipelineSup(tmp_path)
    with pytest.raises(ToolInputError, match="mirror_reference"):
        run(sup=sup, **request(tmp_path, mirror_reference=""))


def test_measuring_with_no_measurement_lists_is_refused(tmp_path):
    sup = PipelineSup(tmp_path)
    with pytest.raises(ToolInputError, match="one measurement list per"):
        run(sup=sup, **request(tmp_path, measurements=""))


def test_a_measurement_folder_missing_a_region_names_which(tmp_path):
    sup = PipelineSup(tmp_path)
    # Built BEFORE the removal: `request` writes the default lists itself, so
    # removing one first would simply be undone.
    arguments = request(tmp_path)
    os.remove(os.path.join(arguments["measurements"], "Measurements_MAND.xlsx"))

    with pytest.raises(ToolInputError) as raised:
        run(sup=sup, **arguments)
    assert "Mandible" in str(raised.value)


def test_registering_with_no_segmentation_bundle_is_refused(tmp_path):
    sup = PipelineSup(tmp_path)
    with pytest.raises(ToolInputError, match="segmentation_model"):
        run(sup=sup, **request(tmp_path, segmentation_model=""))


def test_the_request_is_refused_before_a_single_volume_is_read(tmp_path):
    """A request that cannot work has to come back in a second, not after an
    hour of registration."""
    sup = PipelineSup(tmp_path)
    with pytest.raises(ToolInputError):
        run(sup=sup, **request(tmp_path, measurements=""))
    assert sup.calls == []


def test_no_landmark_bundle_named_is_a_request_not_an_omission(tmp_path):
    """Which weights the landmark tool predicts with is that tool's business.

    ALI_CBCT resolves its own from the deployment's data folder, and ASO --
    which reaches it for the orientation -- asks only for a supervisor. VFACE
    demanded a name anyway, so a panel that had stopped showing the field
    refused every run it sent. AREG_CBCT dropped the same requirement for the
    same reason.
    """
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path, landmark_model=""))
    asked = [params for name, params in sup.calls if name == "ASO"]
    assert asked, "the orientation never ran"
    assert not [params for params in asked if params.get("landmark_model")]


# ---------------------------------------------------------------------------
# The modes that start further down
# ---------------------------------------------------------------------------

def _oriented_cohort(tmp_path):
    """Scans already in the two frames, with the transforms that put them there.

    Both, because the `.tfm` is what lets one frame's landmarks be expressed in
    the other's -- so a caller starting here has to send it, exactly as the
    orientation wrote it.
    """
    from conftest import write_transform

    for patient in PATIENTS:
        for frame in catalogs.FRAMES.values():
            write_volume(tmp_path / "oriented" / f"{patient}_{frame['suffix']}.nii.gz")
            write_transform(tmp_path / "oriented" / f"{patient}_{frame['suffix']}.tfm")
    return str(tmp_path / "oriented")


def test_already_oriented_scans_skip_the_orientation(tmp_path):
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path, mode=catalogs.MODE_ORIENTED,
                           t1=_oriented_cohort(tmp_path)))

    assert sup.asked("ASO") == []
    assert sup.asked("AREG_CBCT")


def test_a_scan_that_does_not_say_its_frame_is_refused_rather_than_guessed(tmp_path):
    """Putting a maxillary scan in the cranial base's folder registers it
    against the wrong masks and reports it as a success."""
    sup = PipelineSup(tmp_path)
    for patient in PATIENTS:
        write_volume(tmp_path / "unlabelled" / f"{patient}_T1.nii.gz")

    with pytest.raises(ToolInputError) as raised:
        run(sup=sup, **request(tmp_path, mode=catalogs.MODE_ORIENTED,
                               t1=str(tmp_path / "unlabelled")))
    assert "CB_Or" in str(raised.value)


# ---------------------------------------------------------------------------
# All the way to a verdict
# ---------------------------------------------------------------------------

def test_a_run_with_a_template_and_a_bundle_writes_a_classification(tmp_path):
    """The whole tool, end to end: two CBCTs in, a verdict per patient out.

    The models are real LightGBM boosters trained in `test_classify.py`'s
    fixture on the same three feature names the template asks for, so what is
    checked is that the numbers travelled -- through the mirror, the
    registration, the measurements, the signed features and into a prediction.
    """
    from test_classify import FEATURES, train

    bundle = tmp_path / "vface_models"
    for name in ("sym_asymm.txt", "mand_asym.txt", "max_asym.txt"):
        train(bundle / name, FEATURES, lambda row: row[FEATURES[0]] > 0)

    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path,
        feature_template=write_feature_template(tmp_path / "template.xlsx", FEATURES),
        classifier_model=str(bundle),
    ))

    written = tree_of(tmp_path / "out")
    assert os.path.join("Measurements", dispatch.FEATURE_TABLE_NAME) in written
    assert os.path.join("Classification", dispatch.CLASSIFICATION_NAME) in written

    verdict = read_table(tmp_path / "out" / "Classification" / dispatch.CLASSIFICATION_NAME)
    assert sorted(verdict["ID"].astype(str)) == ["1", "2"]
    assert set(verdict["Asymmetry"]) <= {"Symmetric", "Asymmetric"}
    assert read_report(tmp_path / "out")["classified"] == 2


def test_the_feature_table_carries_the_measurements_that_were_made(tmp_path):
    """Each column is filled from the region its NAME says, and nowhere else:
    `MAND_RCo_RCo_IS` comes from the mandible's table."""
    from test_classify import FEATURES

    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path,
        feature_template=write_feature_template(tmp_path / "template.xlsx", FEATURES),
    ))

    table = read_table(tmp_path / "out" / "Measurements" / dispatch.FEATURE_TABLE_NAME)
    assert list(table.columns) == ["ID"] + FEATURES
    # ROr sits 28 mm off the midline; the transverse feature is signed, and the
    # skeletal table reads a right-sided landmark moving outwards as Lateral.
    assert abs(table["CB_ROr_ROr_RL"].iloc[0]) == pytest.approx(56.0, abs=1e-6)
    # The mandibular condyle is nudged 1.5 mm superiorly by the fixture.
    assert abs(table["MAND_RCo_RCo_IS"].iloc[0]) == pytest.approx(1.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Heat maps
# ---------------------------------------------------------------------------

def test_heat_maps_are_drawn_per_patient_and_per_region(tmp_path):
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path,
        outputs=catalogs.OUTPUT_VISUALISATION,
        surface_model=str(tmp_path / "models" / "bds"),
    ))

    written = tree_of(tmp_path / "out")
    for region in catalogs.REGIONS:
        for patient in PATIENTS:
            assert os.path.join("Heat maps", region, f"{patient}_{region}_heatmap.vtk") in written
    assert read_report(tmp_path / "out")["heat_maps"] == len(PATIENTS) * len(catalogs.REGIONS)


def test_a_heat_map_run_asks_for_no_landmarks_at_all(tmp_path):
    """Measurements are computed from landmarks and heat maps from surfaces, so
    a deployment missing one tool can still answer the other half."""
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path,
        outputs=catalogs.OUTPUT_VISUALISATION,
        surface_model=str(tmp_path / "models" / "bds"),
    ))
    assert sup.asked("ALI_CBCT") == []
    assert sup.asked("Batch_Dental_Seg")


def test_heat_maps_with_no_surface_bundle_are_refused(tmp_path):
    sup = PipelineSup(tmp_path)
    with pytest.raises(ToolInputError, match="surface_model"):
        run(sup=sup, **request(tmp_path, outputs=catalogs.OUTPUT_VISUALISATION))


def test_asking_for_both_produces_both(tmp_path):
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path,
        outputs=catalogs.OUTPUT_BOTH,
        surface_model=str(tmp_path / "models" / "bds"),
    ))

    written = tree_of(tmp_path / "out")
    assert any(name.startswith("Heat maps") for name in written)
    assert any(name.startswith("Measurements") for name in written)


def test_measuring_someone_elses_registration_needs_it_to_be_supplied(tmp_path):
    """Taking the transforms from `t1` would have read the ORIENTATION matrices
    sitting beside the scans as though they were registrations: the landmarks
    would have been moved by the wrong one, and every measurement would have
    come out of a run that reported success."""
    sup = PipelineSup(tmp_path)
    with pytest.raises(ToolInputError) as raised:
        run(sup=sup, **request(tmp_path, mode=catalogs.MODE_REGISTERED,
                               t1=_oriented_cohort(tmp_path)))
    assert "registration_transforms" in str(raised.value)
    assert sup.calls == []


def test_a_supplied_registration_is_measured_without_registering_again(tmp_path):
    from conftest import write_transform

    oriented = _oriented_cohort(tmp_path)
    for region in catalogs.REGIONS:
        for patient in PATIENTS:
            write_transform(tmp_path / "registrations" / region /
                            f"{patient}_{region}_Reg_transform.tfm")

    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(
        tmp_path, mode=catalogs.MODE_REGISTERED, t1=oriented,
        registration_transforms=str(tmp_path / "registrations"),
    ))

    assert sup.asked("ASO") == []
    assert sup.asked("AMASSS") == []
    assert sup.asked("AREG_CBCT") == []
    assert os.path.join("Measurements", "Measurements_CB.xlsx") in tree_of(tmp_path / "out")


# ---------------------------------------------------------------------------
# The frames a request actually segments
# ---------------------------------------------------------------------------

def test_a_frame_no_region_registers_on_is_not_segmented(tmp_path):
    """Both frames are oriented whatever was asked for -- the landmarks are
    predicted in the cranial base frame and derived into the maxillary one, so
    the measurement path needs both. But a frame no region registers on has no
    structures, and AMASSS rightly refuses an empty list. A cranial-base-only
    request died there, after both orientations had been paid for.
    """
    sup = PipelineSup(tmp_path)
    run(sup=sup, **request(tmp_path, regions=[catalogs.REGION_CRANIAL_BASE]))
    segmented = [params for name, params in sup.calls if name == "AMASSS"]
    assert len(segmented) == 1, f"{len(segmented)} segmentations for one region"
    assert all(params["structures"] for params in segmented)


def test_a_longitudinal_study_orients_its_follow_up_too(tmp_path):
    """The follow-up goes through the same frames as the baseline, and that
    second orientation is a separate code path with no coverage of its own."""
    sup = PipelineSup(tmp_path)
    for patient in PATIENTS:
        write_volume(tmp_path / "t2" / f"{patient}_T2.nii.gz")
    run(sup=sup, **request(
        tmp_path,
        study=catalogs.STUDY_LONGITUDINAL,
        t2=str(tmp_path / "t2"),
    ))
    oriented = [params for name, params in sup.calls if name == "ASO"]
    assert len(oriented) == 4, "two frames, baseline and follow-up"

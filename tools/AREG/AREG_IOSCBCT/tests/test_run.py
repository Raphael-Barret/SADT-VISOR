"""AREG_IOSCBCT's argument rules and its published schema: no GPU, no weights,
no network.

Split out of the single `tools/AREG/tests/test_run.py` AREG had before it
became three tools; see AREG_CBCT/tests/test_run.py for why none of them ran.

The tools this one drives are stood in for by a fake supervisor, which is all a
tool can see of them: five members, duck-typed, nothing imported across venvs.
`FakeSup` and the cohort builders live in `conftest.py`.

Every rule below is checked BEFORE a file is read -- most of these tests pass
paths that do not exist, and reaching the file system would raise something
else entirely. That is the point: a request that cannot work has to come back
in a second, not after an hour of registration.
"""

import inspect
import json
import logging
import os
from types import SimpleNamespace
import typing
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeSup, cohort, run_registration, write_volume
from sadt_areg_ioscbct import dispatch, pipeline, run, tools
from sadt_areg_ioscbct.layout import LAYOUT
from sadt_areg_common import catalogs
from sadt_areg_common.errors import SupervisorRequired, ToolInputError


def _choices(argument):
    """The `Literal` options `run()` publishes for one argument."""
    hint = typing.get_type_hints(run)[argument]
    if typing.get_origin(hint) is list:
        hint = typing.get_args(hint)[0]
    return list(typing.get_args(hint))


def _main(tmp_path=None, **overrides):
    arguments = {
        "ios": "/nonexistent/ios",
        "cbct": "/nonexistent/cbct",
        "output_dir": str(tmp_path / "out") if tmp_path else "/nonexistent/out",
        "automation": catalogs.AUTOMATION_REGISTRATION,
    }
    arguments.update(overrides)
    return dispatch.main(**arguments)


def test_every_tool_is_named_by_string():
    """`sup.run("ASO", ...)`, never `sup.ASO(...)`. A typo in a string is
    greppable and tools.py is the whole call graph; a typo in an attribute is an
    AttributeError an hour into a job.

    Here rather than in AREG_CBCT, which was where the single pre-split suite
    left it: this is the tool that drives all four, so it is the only one whose
    tools.py can be expected to name all four.
    """
    source = open(tools.__file__, encoding="utf-8").read()
    assert 'sup.run("' in source
    for tool in ("Crown_Seg", "ALI_CBCT", "ALI_IOS", "ASO"):
        assert f'"{tool}"' in source, tool


# ---------------------------------------------------------------------------
# Progress -- the waypoints, and the loop after them
# ---------------------------------------------------------------------------

def _events(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def test_each_step_says_which_tool_the_run_is_inside(tmp_path):
    """The three waypoints, which `FakeSup.messages` has recorded and nothing
    read until now.

    They are what a watcher has instead of a frozen bar: a fully-automated run
    spends most of its time inside ALI and ASO, and without them the panel
    cannot say which. They rise, and none of them names a file.
    """
    planted = tmp_path / "planted"
    planted.mkdir()
    sup = FakeSup(tmp_path, {name: (lambda params: planted) for name in
                             ("ALI_CBCT", "ALI_IOS", "ASO")})

    tools.predict_cbct_landmarks(sup, str(tmp_path), "")
    tools.predict_ios_landmarks(sup, str(tmp_path), "")
    tools.orient_cbct(sup, str(tmp_path), str(tmp_path), "")

    fractions = [fraction for fraction, _message in sup.messages]
    assert fractions == [0.1, 0.3, 0.5]
    assert [message for _fraction, message in sup.messages] == [
        "predicting CBCT landmarks with ALI_CBCT",
        "predicting intraoral landmarks with ALI_IOS",
        "orienting the CBCT with ASO",
    ]


def test_the_registration_loop_starts_where_the_waypoints_stopped(tmp_path, monkeypatch):
    """A mode that predicted nothing must not report itself as 60% done.

    The waypoints above only fire in the two modes that call other tools; the
    Registration mode calls none, so its loop owns the whole bar. Getting this
    wrong is invisible in a result and obvious to whoever is watching.
    """
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("SADT_PROGRESS_FILE", str(events_file))
    for patient in ("P1", "P2"):
        # conftest's writer, the suite's own: this test stubs every reader
        # below, so the volume is named and never opened.
        write_volume(tmp_path / "cbct" / f"{patient}_scan.nii.gz")
        (tmp_path / "ios").mkdir(exist_ok=True)
        (tmp_path / "ios" / f"{patient}_Upper.vtk").write_text("")

    # Everything inside the loop is stood in for; the loop itself, which is
    # what reports, runs for real.
    # Keyed PER PATIENT, because the loop narrows landmarks to their own
    # patient now (`_for_patient`): a file name no `patient_key` can read
    # matches nobody in a two-patient batch, which is the point of that rule --
    # it is what stopped patient 2's mesh being registered onto patient 1's
    # landmarks, silently and with an "ok" in the report.
    monkeypatch.setattr(
        dispatch, "_landmarks_by_jaw",
        lambda root: {f"{name}_U.json": {"A": [0.0, 0.0, 0.0]}
                      for name in ("P1", "P2")},
    )
    # A stand-in that answers the one call the loop makes on a mesh. Returning
    # None here made the loop raise on `mesh.transform`, which this test would
    # have reported as a progress failure.
    monkeypatch.setattr(
        dispatch, "_read_mesh",
        lambda path: SimpleNamespace(transform=lambda matrix, inplace=False: None),
    )
    # Contouring the CBCT is the expensive half of a real run and this test is
    # about the progress bar, so it is stood in for like everything else inside
    # the loop.
    monkeypatch.setattr(dispatch, "_cbct_surface", lambda path, sets: (None, {}, None))
    monkeypatch.setattr(
        dispatch.pipeline, "register_one",
        lambda mesh, moving, fixed, surface, on_enamel, max_dist: (
            np.eye(4), {"rms": 0.0}),
    )
    monkeypatch.setattr(
        dispatch, "_write_mesh",
        lambda mesh, path: (
            os.makedirs(os.path.dirname(path), exist_ok=True), open(path, "w").close()
        ),
    )

    for start, expected in ((0.0, [0.0, 0.5]), (0.6, [0.6, 0.8])):
        events_file.write_text("")
        report = {"patients": {}}
        dispatch.register(
            ios_dir=str(tmp_path / "ios"), cbct_dir=str(tmp_path / "cbct"),
            ios_landmark_dir=str(tmp_path), cbct_landmark_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"), suffix="Reg", report=report,
            max_dist=1.0, progress_start=start,
        )
        events = _events(events_file)
        assert [event["message"] for event in events] == ["patient 1 of 2", "patient 2 of 2"]
        assert [event["fraction"] for event in events] == expected


# ---------------------------------------------------------------------------
# What reaches the log: how many, never which
# ---------------------------------------------------------------------------

def test_the_unpaired_patients_are_counted_in_the_log_not_named(tmp_path, caplog):
    """A patient key is the caller's own file name -- and here it can BE the
    file's stem: `_patient_key` falls back to it when a name holds no digit.

    A tool's stderr is captured to a file in the job directory, and on a FAILED
    run the server copies its tail into its own persistent log, so a key
    written here outlives the run. The returned mapping still names every one
    of them, and the run report carries it back to whoever sent the data.
    """
    (tmp_path / "ios").mkdir()
    (tmp_path / "cbct").mkdir()
    (tmp_path / "ios" / "P001_Upper.vtk").write_text("")
    (tmp_path / "cbct" / "P001_scan.nii.gz").write_text("")
    (tmp_path / "ios" / "Smith_John_Upper.vtk").write_text("")

    with caplog.at_level(logging.INFO, logger="sadt_areg_ioscbct.pipeline"):
        paired, unpaired = pipeline.discover(str(tmp_path / "ios"), str(tmp_path / "cbct"))

    assert list(paired) == ["1"]
    assert "Smith_John_Upper" in unpaired, "the caller is still told which"

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Not registered, only one modality present: 1 patient(s) with no CBCT, "
        "0 with no intraoral scan"
    ], messages
# The modes
# ---------------------------------------------------------------------------

def test_the_default_mode_names_no_mode_at_all():
    """It used to be `Registration` -- the one that needs no other tool -- so a
    default would not read as a broken tool. But a mode a caller DECLARES can
    disagree with the folder they sent, and a request holding only scans was
    then refused for missing the landmarks `Registration` requires.

    The default now says "read it off the request", and naming a mode is an
    override. See `catalogs.AUTOMATION_AUTO`."""
    assert inspect.signature(run).parameters["automation"].default == (
        catalogs.AUTOMATION_AUTO
    )
    assert catalogs.AUTOMATION_AUTO not in (
        catalogs.AUTOMATION_REGISTRATION,
        catalogs.AUTOMATION_SEMI,
        catalogs.AUTOMATION_FULLY,
    )


def test_the_published_modes_are_the_catalog_s_own_for_this_modality():
    """`Literal` takes literals only, so the signature is a second declaration
    of the same set. A mode offered here and refused by `main` is a picker that
    produces 422s; one `main` accepts and the signature omits is a mode no
    client can reach.

    The ORDER differs on purpose -- the signature leads with the default, so a
    client's picker opens on the value that asks nothing of the reader."""
    assert set(_choices("automation")) == set(
        catalogs.AUTOMATION_BY_MODALITY[catalogs.MODALITY_IOSCBCT]
    )
    assert _choices("automation")[0] == catalogs.AUTOMATION_AUTO


def test_the_oriented_mode_belongs_to_the_cbct_tool_alone(tmp_path):
    """Orienting before registering is a CBCT step, and there is nothing to
    orient an intraoral scan onto in this mode."""
    assert catalogs.AUTOMATION_ORIENTED not in _choices("automation")
    with pytest.raises(ToolInputError) as raised:
        _main(tmp_path, automation=catalogs.AUTOMATION_ORIENTED)
    assert catalogs.AUTOMATION_ORIENTED in str(raised.value)


def test_an_unknown_mode_names_the_value_and_what_is_offered(tmp_path):
    """`Literal` is published, not enforced -- the runner calls `run(**params)`
    from a JSON object -- so a stale client has to be told."""
    with pytest.raises(ToolInputError) as raised:
        _main(tmp_path, automation="Magic")
    message = str(raised.value)
    assert "'Magic'" in message
    for mode in catalogs.AUTOMATION_BY_MODALITY[catalogs.MODALITY_IOSCBCT]:
        assert mode in message


def test_an_omitted_mode_is_read_off_the_request(tmp_path):
    """`main` is also called directly by tests and by another tool, where the
    signature's default does not apply -- so `None` has to mean the same thing
    the default means.

    It used to fall back to `Registration`, which then refused a request holding
    only scans for missing the landmarks it requires. A request holding only
    scans is now what it looks like: one with everything still to predict."""
    assert dispatch.derive_automation(None, None, None) == (
        catalogs.AUTOMATION_FULLY, "from the data"
    )
    with pytest.raises(ToolInputError) as raised:
        _main(tmp_path, automation=None)
    assert "Registration mode takes" not in str(raised.value)


def test_both_landmark_sets_mean_there_is_nothing_left_to_predict():
    mode, source = dispatch.derive_automation(None, "/ios/lm", "/cbct/lm")

    assert mode == catalogs.AUTOMATION_REGISTRATION
    assert source == "from the data"


def test_one_landmark_set_alone_is_not_registration_mode():
    """Half of the pair is not a smaller answer, it is no answer: the
    cross-modality registration needs points on both sides."""
    assert dispatch.derive_automation(None, "/ios/lm", None)[0] != (
        catalogs.AUTOMATION_REGISTRATION
    )
    assert dispatch.derive_automation(None, None, "/cbct/lm")[0] != (
        catalogs.AUTOMATION_REGISTRATION
    )


def test_orienting_the_cbct_first_is_what_separates_the_two_predicted_modes():
    """The one thing here no folder can answer, so it is asked as itself rather
    than hidden inside a three-valued mode."""
    assert dispatch.derive_automation(None, None, None, orient_cbct_first=True)[0] == (
        catalogs.AUTOMATION_FULLY
    )
    assert dispatch.derive_automation(None, None, None, orient_cbct_first=False)[0] == (
        catalogs.AUTOMATION_SEMI
    )


def test_a_named_mode_overrides_what_the_request_looks_like():
    mode, source = dispatch.derive_automation(
        catalogs.AUTOMATION_SEMI, "/ios/lm", "/cbct/lm"
    )

    assert mode == catalogs.AUTOMATION_SEMI
    assert source == "requested"


# ---------------------------------------------------------------------------
# What each mode needs
# ---------------------------------------------------------------------------

def test_registration_mode_without_landmarks_names_both_arguments(tmp_path):
    with pytest.raises(ToolInputError) as raised:
        _main(tmp_path)
    message = str(raised.value)
    assert "ios_landmarks" in message and "cbct_landmarks" in message


def test_registration_mode_with_only_one_side_is_still_refused(tmp_path):
    with pytest.raises(ToolInputError, match="Registration mode takes"):
        _main(tmp_path, ios_landmarks="/nonexistent/lm")
    with pytest.raises(ToolInputError, match="Registration mode takes"):
        _main(tmp_path, cbct_landmarks="/nonexistent/lm")


def test_registration_mode_needs_no_supervisor_at_all(tmp_path):
    """This is what makes the tool usable standalone: the mode that predicts
    nothing calls nothing."""
    cohort(tmp_path)
    assert run_registration(tmp_path, sup=None)


@pytest.mark.parametrize(
    "automation", [catalogs.AUTOMATION_SEMI, catalogs.AUTOMATION_FULLY]
)
def test_an_automated_mode_without_a_supervisor_refuses_at_the_door(tmp_path, automation):
    """Checked up front, before a single file is read. Crown_Seg is the first
    of the three, so it is the one named."""
    with pytest.raises(SupervisorRequired) as raised:
        _main(tmp_path, automation=automation)
    assert "Crown_Seg" in str(raised.value)
    assert automation in str(raised.value)


def test_the_rules_run_before_anything_is_read():
    """Every case above passes paths that do not exist."""
    assert not os.path.exists("/nonexistent/ios")


def test_an_unreadable_input_is_reported_rather_than_crashing_the_batch(tmp_path):
    """`discover` walks a folder that is not there without complaining, so the
    refusal is the pairing one -- which names what it was looking for."""
    cohort(tmp_path)
    with pytest.raises(ToolInputError, match="No patient has both"):
        run_registration(tmp_path, cbct="/nonexistent/cbct")


# ---------------------------------------------------------------------------
# max_dist
# ---------------------------------------------------------------------------

def _captured_max_dist(tmp_path, monkeypatch, **overrides):
    from sadt_areg_ioscbct import pipeline

    seen = []
    real = pipeline.register_one

    def spy(*args, **kwargs):
        seen.append(kwargs["max_dist"])
        return real(*args, **kwargs)

    monkeypatch.setattr(dispatch.pipeline, "register_one", spy)
    cohort(tmp_path)
    run_registration(tmp_path, **overrides)
    return seen


def test_a_max_dist_of_zero_means_upstreams_own_value(tmp_path, monkeypatch):
    """0 is what an untouched numeric field sends, and it is not a distance
    anybody meant -- every point would be further from its neighbour than
    that, and the ICP would match nothing.

    1.0, not the 1.5 in upstream's `run_icp_point_to_plane` signature: every
    call site upstream has passes 1.0, and the call sites are what produced the
    results this tool is checked against.
    """
    assert _captured_max_dist(tmp_path, monkeypatch, max_dist=0.0) == [1.0]


def test_an_omitted_max_dist_means_the_same(tmp_path, monkeypatch):
    assert _captured_max_dist(tmp_path, monkeypatch) == [1.0]


@pytest.mark.parametrize("given", [0.25, 1.5, 12.0])
def test_a_max_dist_the_caller_named_is_the_one_used(tmp_path, monkeypatch, given):
    assert _captured_max_dist(tmp_path, monkeypatch, max_dist=given) == [given]


def test_an_integer_max_dist_is_accepted_as_a_distance(tmp_path, monkeypatch):
    """A form field sends `3`, not `3.0`, and the ICP compares it against
    floats."""
    assert _captured_max_dist(tmp_path, monkeypatch, max_dist=3) == [3.0]


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

# The arguments the SERVER adds for a tool that calls another, which a layout
# is allowed to name and `run()` therefore does not take. Restated here because
# `scripts/describe.py` is a script at the root of the repository and not an
# importable package -- the same way ALI_IOS's suite restates it.
INJECTED_ARGUMENTS = ("keep_intermediate", "stop_after")


def test_the_layout_only_names_arguments_the_signature_offers():
    """`describe.py` refuses a hint naming an argument `run()` does not take
    AND is not injected, so a stale entry here is a tool that will not publish
    its schema at all."""
    allowed = set(inspect.signature(run).parameters) | set(INJECTED_ARGUMENTS)
    assert set(LAYOUT) <= allowed


def test_every_layout_condition_names_a_mode_this_tool_has():
    """A `visible_when` naming a mode that does not exist hides the field for
    good, and a client cannot tell that from a field the tool never
    declared."""
    offered = set(catalogs.AUTOMATION_BY_MODALITY[catalogs.MODALITY_IOSCBCT])
    for argument, hints in LAYOUT.items():
        condition = hints.get("visible_when")
        if not condition:
            continue
        assert set(condition) <= set(inspect.signature(run).parameters), argument
        wanted = condition["automation"]
        for mode in (wanted if isinstance(wanted, list) else [wanted]):
            assert mode in offered, (argument, mode)


def test_the_landmark_folders_are_hidden_in_the_modes_that_overwrite_them():
    """Showing them in a mode that predicts its own is how a user comes to
    believe their files were used.

    The auto value is in the condition beside Registration, and has to be: it is
    the default now, and filling these folders is HOW Registration is selected.
    A condition naming Registration alone would have hidden the two fields that
    select Registration."""
    for argument in ("ios_landmarks", "cbct_landmarks"):
        shown_in = LAYOUT[argument]["visible_when"]["automation"]
        assert catalogs.AUTOMATION_REGISTRATION in shown_in
        assert catalogs.AUTOMATION_AUTO in shown_in
        assert catalogs.AUTOMATION_SEMI not in shown_in
        assert catalogs.AUTOMATION_FULLY not in shown_in


def test_the_orientation_reference_is_shown_only_in_the_mode_that_orients():
    shown_in = LAYOUT["cbct_reference"]["visible_when"]["automation"]

    assert catalogs.AUTOMATION_FULLY in shown_in
    assert catalogs.AUTOMATION_AUTO in shown_in
    assert catalogs.AUTOMATION_REGISTRATION not in shown_in


def test_the_one_choice_no_folder_can_answer_is_asked_as_itself():
    """`orient_cbct_first` replaces what used to be the difference between two
    values of `automation`. A three-valued mode was the wrong shape: two of its
    values were facts about the files, the third was a preference."""
    # The label names the FRAME: "orient first" said what the tool does, not
    # what the reader gets, and a clinician cannot act on the difference.
    label = LAYOUT["orient_cbct_first"]["label"]
    assert "Frankfurt" in label
    # Spelled like the bundle and like the rest of the repository, never the
    # anatomical "Frankfort" -- a panel and a folder differing by a letter is a
    # support question.
    assert "Frankfort" not in label
    assert not LAYOUT["orient_cbct_first"].get("hidden")
    shown_in = LAYOUT["orient_cbct_first"]["visible_when"]["automation"]
    assert catalogs.AUTOMATION_REGISTRATION not in shown_in


def test_the_mode_itself_is_not_put_to_the_reader():
    assert LAYOUT["automation"]["hidden"] is True


def test_every_published_argument_has_a_label(tmp_path):
    """The client builds its panel from these; an argument with none falls back
    to its identifier, which is how a panel ends up saying `cbct_landmarks`
    above a file picker."""
    published = [
        name for name in inspect.signature(run).parameters
        # `sup` and `data_root` are INJECTED by the server (describe.INJECTED),
        # never published and never laid out: a panel cannot offer a supervisor
        # or the server's own data folder.
        if name not in ("output_dir", "sup", "data_root")
    ]
    # The injected ones are not published by this tool and carry no label: they
    # are named here only to be hidden, which is presentation and not an
    # argument of `run()`.
    assert sorted(set(LAYOUT) - set(INJECTED_ARGUMENTS)) == sorted(published)
    for name in published:
        assert LAYOUT[name].get("label"), name


def test_the_two_inputs_are_named_for_the_modality_not_for_a_timepoint():
    """NOT longitudinal, unlike its two siblings: one timepoint imaged two
    ways, which is why the arguments are `ios` and `cbct` rather than `t1` and
    `t2`."""
    parameters = set(inspect.signature(run).parameters)
    assert {"ios", "cbct"} <= parameters
    assert not {"t1", "t2"} & parameters


def test_the_supervisor_is_keyword_only():
    """It is injected by the runner, never sent by a client, and
    `describe.py` publishes only what precedes the bare `*`."""
    signature = inspect.signature(run)
    assert signature.parameters["sup"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sup"].default is None


def test_a_supervisor_is_accepted_and_unused_in_registration_mode(tmp_path):
    """A server always supplies one. The mode that needs nothing must not start
    calling something because it was given the means to."""
    cohort(tmp_path)
    sup = FakeSup(tmp_path)
    run_registration(tmp_path, sup=sup)
    assert sup.calls == []

"""How a client should lay this tool's panel out. Presentation only."""

from sadt_areg_common import catalogs

_INPUTS = "Inputs"
# "Advanced", not "Landmarks", and the name does two things at once.
#
# It is what the section IS: landmarks a caller already has are the escape hatch
# from predicting them, which is the same role ASO's landmark folder plays and
# the reason ASO puts it under this name. The client folds a section called
# "Advanced" shut by default (base_widget._COLLAPSED_SECTIONS), so someone who
# does not have landmarks no longer steps over two file pickers on the way to
# Apply.
#
# And it fixes where the box SITS. `formgen.sections_of` orders sections by
# first mention in the schema, and the facade composes AREG_CBCT first -- so a
# section only this engine named was first mentioned after AREG_CBCT's
# "Outputs", and the panel showed Landmarks BELOW the output folder. Under this
# name it merges into the Advanced box AREG_CBCT already declares, which comes
# before Outputs.
_LANDMARKS = "Advanced"
_MODELS = "Models"
_OUTPUTS = "Outputs"

# Registration takes the landmarks; the other two predict them. Showing the
# landmark folders in a mode that overwrites them is how a user comes to believe
# their files were used.
#
# `AUTOMATION_AUTO` is in all three because it is now the DEFAULT value of
# `automation`, which names no mode: a condition listing only the named modes
# would match nothing on a request nobody has overridden, and every field it
# guards would vanish -- including the landmark folders, which are how
# Registration mode is SELECTED. In the auto case the reader sees the superset
# and picks a mode by filling a field; someone who overrides sees only what
# their mode reads.
_SUPPLIED = {"automation": [catalogs.AUTOMATION_AUTO,
                            catalogs.AUTOMATION_REGISTRATION]}
_PREDICTED = {"automation": [catalogs.AUTOMATION_AUTO, catalogs.AUTOMATION_SEMI,
                             catalogs.AUTOMATION_FULLY]}
_ORIENTED = {"automation": [catalogs.AUTOMATION_AUTO, catalogs.AUTOMATION_FULLY]}

LAYOUT = {
    # Injected by the server for every tool that calls another (see
    # describe.INJECTED_ARGUMENTS), and unnamed it arrives with a generic label
    # in a section of its own -- an "Intermediate results" box in a panel that
    # never asked for one. Hidden rather than renamed: what this chain leaves
    # behind on the way is not something a clinician is being offered yet, and
    # a check box that promises files nobody has decided to return is worse
    # than no check box.
    "keep_intermediate": {"hidden": True},

    # Injected by the server too, from the checkpoints this chain offers, and
    # it lands in a "Quality control" section of its own. Not offered yet:
    # stopping a run for review is a workflow this deployment has not decided
    # on, and a box that stops a cohort halfway is not one to leave lying
    # around until it has.
    "stop_after": {"hidden": True},

    "ios": {"section": _INPUTS, "label": "Intraoral scans"},
    "cbct": {"section": _INPUTS, "label": "CBCT volumes"},
    # NOT "Mode". The facade publishes its own `mode` -- the modality -- and
    # labels it "Mode" too, so a panel reached through AREG shows two dropdowns
    # side by side under the same word: one saying CBCT, the next saying
    # Semi-Automated. A user who read the first as "the mode" then hunted for
    # check boxes that only exist in another value of the SECOND one, and had
    # no way to tell which was which. The engines keep this label when opened
    # directly, where there is only one dropdown and no ambiguity, so the word
    # has to carry its own meaning either way.
    # Read off the request (see dispatch.derive_automation): both landmark sets
    # supplied is Registration, which predicts nothing. Still an argument, as an
    # override.
    #
    # What it CANNOT derive is the difference between the other two -- orienting
    # the CBCT first is a clinical decision, not a property of a folder -- so
    # that is asked below as `orient_cbct_first`, a box that says what it does.
    # A three-valued mode was the wrong shape for it: two of its values were
    # facts about the files and the third was a preference.
    #
    # Hidden here as well as in the other two engines: the facade publishes ONE
    # `automation` and takes its presentation from whichever engine composes
    # first, so a hint left in one place would depend on the order of a dict in
    # deployment.toml.
    "automation": {"section": _INPUTS, "label": "Automation", "hidden": True},
    # The one choice this mode really has. Visible, and only where it is read:
    # with both landmark sets in hand there is nothing to predict and nothing to
    # orient for.
    # The label names the frame, because "orient first" said what the tool does
    # and not what the reader gets. Spelled "Frankfurt" like the bundle and like
    # the rest of this repository, not the anatomical "Frankfort": a panel and a
    # folder disagreeing by a letter is a support question.
    #
    # A yes/no here, unlike AREG_CBCT's `orientation`, and that difference is the
    # point: there only ONE frame is valid in this mode, the landmarks the rest
    # of the chain reads being defined in it, so offering the occlusal plane
    # would offer a way to be silently wrong.
    "orient_cbct_first": {
        "section": _INPUTS,
        "label": "Standardise the CBCT orientation (Frankfurt)",
        "visible_when": _PREDICTED,
    },

    "ios_landmarks": {
        "section": _LANDMARKS, "label": "Intraoral landmarks", "visible_when": _SUPPLIED,
    },
    "cbct_landmarks": {
        "section": _LANDMARKS, "label": "CBCT landmarks", "visible_when": _SUPPLIED,
    },

    # Not offered: Crown_Seg finds its own weights, so there is nothing here for a
    # clinician to decide and a wrong pick is a chain that predicts with the
    # wrong weights.
    "crown_model": {
        "hidden": True, "section": _MODELS, "label": "Crown segmentation model", "visible_when": _PREDICTED,
    },
    # Not offered: ALI_IOS finds its own, so there is nothing here for a
    # clinician to decide and a wrong pick is a chain that predicts with the
    # wrong weights.
    "ios_landmark_model": {
        "hidden": True, "section": _MODELS, "label": "Intraoral landmark bundle", "visible_when": _PREDICTED,
    },
    # Not offered: ALI_CBCT finds its own, so there is nothing here for a
    # clinician to decide and a wrong pick is a chain that predicts with the
    # wrong weights.
    "landmark_model": {
        "hidden": True, "section": _MODELS, "label": "CBCT landmark bundle", "visible_when": _PREDICTED,
    },
    # Not offered: one frame the chain expects (dispatch._own_reference), so there is nothing here for a
    # clinician to decide and a wrong pick is a chain that predicts with the
    # wrong weights.
    "cbct_reference": {
        "hidden": True, "section": _MODELS, "label": "Orientation reference", "visible_when": _ORIENTED,
    },

    # Not offered, and no longer for the reason it once was: this used to tune
    # an ICP that never ran, and now it tunes the one that does. Still hidden,
    # because 1.0 mm is not a clinical choice -- it is the capture radius the
    # whole chain was measured at, upstream included, and a clinician reading
    # "ICP match distance" has no way to know that widening it lets the arch
    # match the opposing one.
    "max_dist": {"section": _OUTPUTS, "label": "ICP match distance (mm)", "hidden": True},
    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}

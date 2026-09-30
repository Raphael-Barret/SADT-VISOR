"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument the signature
does not take. Delete this file and the tool still works; the panel gets worse.

No `modality` condition anywhere, and that is the split showing through. The
merged AREG carried one on almost every field, because two modalities and three
automation modes shared a single schema and most arguments applied to exactly
one combination. There is no other modality here now, so what remains are the
conditions that are really about the MODE.
"""

from sadt_areg_common import catalogs

_INPUTS = "Inputs"
_REGISTRATION = "Registration"
# Where the two inputs that SELECT a mode live, for the reason ASO puts its
# landmark folder there: someone who already has masks, or who wants the T1
# oriented first, goes looking for the field; someone who does not should not
# have to step over it on the way to Apply.
_ADVANCED = "Advanced"
_OUTPUTS = "Outputs"

# Each mirrors a check in dispatch.py. An argument the chosen mode never reads
# is not merely noise: shown as optional beside the ones that matter, it reads
# as something the user chose not to fill, and the refusal then arrives at the
# end of a run instead of before it.
#
# **`AUTOMATION_AUTO` is in every one of them, and leaving it out would empty the
# panel.** The mode is derived now, so the default value of `automation` names no
# mode -- a condition listing only Fully-Automated and Oriented would match
# nothing on a request nobody has overridden, and the fields it guards would be
# invisible in the one case that matters. With the auto value in, a clinician who
# overrides still sees only what their mode reads, and everyone else sees the
# superset and selects the mode by filling a field.
_SEGMENTED = {  # the modes that produce their own masks
    "automation": [catalogs.AUTOMATION_AUTO,
                   catalogs.AUTOMATION_FULLY, catalogs.AUTOMATION_ORIENTED]
}
_ORIENTED = {"automation": [catalogs.AUTOMATION_AUTO, catalogs.AUTOMATION_ORIENTED]}
_SEMI = {"automation": [catalogs.AUTOMATION_AUTO, catalogs.AUTOMATION_SEMI]}

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

    "t1": {"section": _INPUTS, "label": "T1 (baseline)"},
    "t2": {"section": _INPUTS, "label": "T2 (follow-up)"},
    # NOT "Mode". The facade publishes its own `mode` -- the modality -- and
    # labels it "Mode" too, so a panel reached through AREG shows two dropdowns
    # side by side under the same word: one saying CBCT, the next saying
    # Semi-Automated. A user who read the first as "the mode" then hunted for
    # check boxes that only exist in another value of the SECOND one, and had
    # no way to tell which was which. The engines keep this label when opened
    # directly, where there is only one dropdown and no ambiguity, so the word
    # has to carry its own meaning either way. The label survives being hidden:
    # a report prints it, and so does the schema.
    #
    # Derived from what the request supplies (see dispatch.derive_automation):
    # masks are Semi-Automated, none is Fully-Automated, and the frame named in
    # `orientation` below picks the oriented variant -- the one thing here no
    # folder can answer. Asking the clinician to declare it meant the answer could
    # disagree with the folder -- and when it did, the folder won, in silence,
    # which is the hole ASO closed first and this is the same close.
    #
    # Still an argument, as the one override it can express: "Fully-Automated"
    # beside a folder of masks means "segment anyway, I know they are there".
    #
    # Hidden in all three engines, not only here: the facade publishes ONE
    # `automation` and takes its presentation from the first engine composed, so
    # a hint left in one place and not the others would depend on the order of a
    # dict in deployment.toml.
    "automation": {"section": _INPUTS, "label": "Automation", "hidden": True},
    # Detected from the data (see dispatch._run_cbct), so the panel does not put
    # the question to a clinician who cannot see the answer either: DICOM slices
    # routinely carry no extension. Still an argument, as the one override it
    # can express -- "convert it anyway".
    "dicom_input": {"section": _INPUTS, "label": "Input is DICOM", "hidden": True},
    # The one choice the automated path really has, asked by NAME rather than as
    # a yes/no: the two published frames mean different things, and a boolean
    # could only have offered the default one. Naming a frame is also what
    # selects the oriented mode, so this replaces both a checkbox and the hidden
    # `reference` it would have needed.
    #
    # Shown only where it is read: with masks in hand there is nothing to segment
    # and nothing to orient for. The long explanation is the argument's
    # `description`, which the client shows as the label's tooltip.
    "orientation": {
        "section": _INPUTS, "label": "Orientation",
        "visible_when": _SEGMENTED,
    },

    # The one argument a clinician must actually think about: register on what
    # has NOT changed between the two timepoints.
    #
    # Chips, the same as `segmentations` below and as AMASSS's `structures`:
    # three lists of anatomy in one panel should read alike, and chips wrap where
    # `inline` did not. No `groups` -- three options are not two kinds of thing,
    # and a heading over each would say less than the chips do.
    #
    # `option_help` gives each region its code, which is the name of the folder
    # the transforms come back in (`CB/C_0001_CB_Reg.nii.gz`), so the panel says
    # where to look for the answer.
    "regions": {
        "section": _REGISTRATION,
        "label": "Register on",
        "ui": "chips",
        "option_help": dict(catalogs.REGION_CODES),
    },
    # In Advanced, and no longer conditioned on a mode the user picks: filling
    # this IS how Semi-Automated is chosen. A condition naming only
    # Semi-Automated would have hidden the field that selects Semi-Automated.
    "t1_masks": {"section": _ADVANCED, "label": "T1 masks", "visible_when": _SEMI},
    # The original's second group, under its own name, and presented the way
    # AMASSS presents the very same list -- chips under the same two headings,
    # in the same order. These structures ARE AMASSS's: a clinician who ticks
    # them there and then here should be looking at one list, not two spellings
    # of it.
    #
    # Chips rather than `inline`, and that is what the earlier note here was
    # really asking for: six labels of "Cervical vertebra" length do not fit
    # across a 425 px panel on one row, and inline does not wrap -- it pushed
    # the section wider than the module panel. Chips wrap. Stacking them one
    # per line was the workaround, not the answer.
    #
    # `option_help` gives each one its AMASSS code, which is the suffix of the
    # file that comes back (`..._seg_MAND.nii.gz`), so the panel says what you
    # will receive rather than only what you asked for. No `min_selected`,
    # unlike AMASSS: none ticked is the right default for a registration run,
    # where AMASSS with no structure would have nothing to do at all.
    "segmentations": {
        "section": _REGISTRATION, "label": "AMASSS segmentation",
        "visible_when": _SEGMENTED,
        "ui": "chips",
        "groups": catalogs.SEGMENTATION_GROUPS,
        "option_help": dict(catalogs.SEGMENTATION_CODES),
    },
    # Not offered: the modes that segment segment with AMASSS, and the bundle
    # is the one the deployment already publishes (see dispatch._own_segmentation).
    # The argument stays, so a caller with a reason can still name another one.
    "segmentation_model": {
        "section": _REGISTRATION, "label": "Segmentation model",
        "visible_when": _SEGMENTED, "hidden": True,
    },
    # `_SEMI`, not `_SEGMENTED`, and the condition was inverted until
    # 2026-09-29. The modes that segment ask AMASSS for `merge=["SEPARATE"]` --
    # one BINARY file per structure -- so there is exactly one non-zero label
    # and 0, "use the whole mask", is the only correct answer; any other value
    # makes `apply_mask` refuse the run. The field was therefore shown in the
    # two modes where it can only do harm, and hidden in the one mode where a
    # clinician supplies their own `t1_masks` and a multi-label segmentation is
    # the case it exists for.
    #
    # Narrow even there, and worth knowing before offering it: `discover_masks`
    # only matches a mask whose name carries its region's token, so a single
    # multi-label file called `P1_seg.nii.gz` is never paired in the first
    # place. What this reaches is a per-region file that happens to hold
    # several labels.
    "segmentation_label": {
        "section": _ADVANCED, "label": "Mask label value", "visible_when": _SEMI,
    },
    # Not offered, for the reason `segmentation_model` gives about its own, plus
    # one: this argument is `server_selectable = "model"`, so a request that does
    # not name a bundle arrives holding `DATA/AREG/models/` -- the FOLDER, eight
    # bundles deep -- which ASO cannot orient onto. The dropdown therefore
    # offered a list where every entry but one was wrong.
    #
    # `dispatch._own_reference` answers instead: there is one frame the oriented
    # mode means, the Frankfort horizontal and the mid-sagittal plane, and
    # picking another silently changes what the registration is expressed in.
    # The argument stays, so a caller with a reason can still name one.
    "reference": {
        "hidden": True,
        "section": _ADVANCED, "label": "Orientation reference",
        "visible_when": _ORIENTED,
    },
    # Not offered: ALI_CBCT resolves its own weights from the deployment's data
    # folder, so there is no second answer here -- and two engines of this
    # facade disagreeing about whether this argument is a hosted NAME or a path
    # is what stopped `AREG` composing at all.
    "landmark_model": {
        "section": _REGISTRATION, "label": "Landmark model bundle",
        "visible_when": _ORIENTED, "hidden": True,
    },

    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}

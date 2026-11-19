# Provenance

Where each tool's algorithm came from. Upstream history is deliberately **not**
grafted into this repository: it is a single history covering sixteen unrelated
modules, and merging it would make neither history readable. This table is the
record instead, and it matters more here than commit history does -- it is what
tells you whether a result came from upstream code or from something we changed.

Upstream is
[DCBIA-OrthoLab/SlicerAutomatedDentalTools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools)
unless a row says otherwise. Each tool's own README carries the same
information in full, including the pins kept and the changes made.

| Tool | Upstream path | Upstream commit | Ported | Algorithm modified |
|---|---|---|---|---|
| [Surg_Mov_Pred](tools/Surg_Mov_Pred/) | `SurgMovPred_CLI/SurgMovPred_CLI.py` | `d7702ae` (2026-06-24) | 2026-08-12 | no -- repackaging only. Model load order sorted for reproducibility, both result tables returned instead of one; predictions bit-identical to the pre-port implementation. |
| [AMASSS](tools/AMASSS/) | `AMASSS_CLI/` | `21a62a8` (2026-05-22) | 2026-08-12 | no -- repackaging only. Pinned to the deployed stack (torch 2.8.0+cu128, nnunetv2 2.8.1) rather than upstream's declared torch 2.2.0 / nnunetv2 2.8.0; masks bit-identical to the pre-port implementation, within nnUNet's own CUDA nondeterminism. |
| [ASO](tools/ASO/) | `ASO/`, `ASO_CBCT/{PRE,SEMI}_ASO_CBCT/`, `ASO_IOS/{PRE,SEMI}_ASO_IOS/` | **unrecorded** -- see below | 2026-08-13 | no -- repackaging only. Nothing is pinned upstream; pinned to the imaging stack the sibling tools lock. Fully-automated CBCT reaches the landmark tool through the supervisor at the point it always ran, so the order is unchanged. Driven end to end through a real supervisor on the real bundle and a real card; not yet diffed numerically against the pre-port ASO. |
| [ALI](tools/ALI/) | `ALI_CBCT/`, `ALI_CBCT_utils/`, `ALI_IOS/`, `ALI_IOS_utils/` | **unrecorded** -- see below | 2026-08-13 | no -- repackaging only. Pinned to the deployed stack (torch 2.8.0+cu128, monai 1.6.0, itk 5.4.7). One visible behaviour change: an unlabelled IOS mesh is refused naming `Crown_Seg` instead of being segmented in-process. CBCT landmarks **bit-identical** to the pre-port implementation on a real scan (0.0000 mm across 16/16 run pairs, both sides deterministic); the IOS half is unvalidated, pytorch3d needing a CUDA toolkit. Mucogingival (a third IOS network, mandible only) added from the server's unmerged `AREG` branch, where it had been written against the in-process ALI. IOS crown networks validated on a real mesh; MG's own predictions are not, for want of a lower arch. |
| [AREG](tools/AREG/) | `AREG/` and its CLI modules, by way of the server's unmerged `AREG` branch | **unrecorded** -- see below | 2026-08-14 | no -- repackaging only. Pinned to the deployed stack, plus `itk-elastix` which no sibling needs. Drives four tools (AMASSS, ASO, Crown_Seg, ALI) through the supervisor, where the in-process version used `registry.TOOLS`. CBCT engine validated end to end against a known transform; the IOS engine and any comparison with the pre-port implementation are **not**. |
| [Crown_Seg](tools/Crown_Seg/) | -- (written against `shapeaxi`; the engine is now vendored) | shapeaxi 2.0.2 helpers, by AST; `iosseg` (lab, ssh3) for the approach | 2026-11-19 | method unchanged, engine replaced: shapeaxi and pytorch3d are no longer called, rendering is the Embree renderer ALI_IOS ships. On the published test scan, CPU: 99.32% of points agree with the pytorch3d run (98.57% over the tooth union, per-tooth Dice 0.963 to 0.999), and two runs are bit-identical. See tools/Crown_Seg/README.md. |
| [Batch_Dental_Seg](tools/Batch_Dental_Seg/) | `BATCHDENTALSEG/BATCHDENTALSEGLib/SegmentationWidget.py` | `6df3fab` (2026-08-05) | 2026-08-12 | no -- repackaging only. Same stack as AMASSS (torch 2.8.0+cu128, nnunetv2 2.8.1); labels compared against the pre-port implementation. |
| [VFACE](tools/VFACE/) | `VFACE/`, `VFACE_utils/`, `VFACE_CLI/`, plus `MRI2CBCT_CLI/MRI2CBCT_CLI_utils/resample.py` | `84ef432` (2026-09-18) | 2026-09-21 | no -- repackaging only, and checked: `tests/test_matches_upstream.py` transcribes upstream's distance, angle and sign-meaning code line for line and drives it against the port, which agrees on every one of ~91 000 random cases. Runs no network of its own. **Two things are not pure repackaging and are named below.** |

A row is filled in by the PR that migrates the tool, in the same commit that
adds the package. "Algorithm modified" is `no` for a pure repackaging and
otherwise names what changed and why, matching the tool README.

**AREG's history does not follow, unlike every other tool's.** It sat on an
unmerged branch when the `git subtree split` carried the tools into this
repository, so `git log --follow` stops at the commit that copied its files in.
The server-side source is preserved at the `archive/AREG` tag in
`VISOR-serve`, and parked there as `server/tools/_AREG/`.

**ALI's, ASO's and AREG's upstream commits are unrecorded, and that is a gap,
not a style.** Both server-side ports landed with no upstream revision in the commit
message and none in the tree -- ALI as `ADD ALI & CrownSeg` (`a0ed474`,
2026-07-31) -- so which upstream commit each algorithm came from cannot be
recovered from either repository. The per-module mappings are exact and are in
[tools/ALI/README.md](tools/ALI/README.md) and
[tools/ASO/README.md](tools/ASO/README.md); only the revisions are missing, and
they need filling in by whoever made those ports. Until then, "no -- repackaging
only" is a claim about code that cannot be pointed at.

**VFACE carries a copy of another tool's code, and that is deliberate.** Its
full pipeline begins by calling `MRI2CBCT`'s resample CLI, and MRI2CBCT is not a
tool here. What VFACE asks it for is not MRI2CBCT: every argument it sends turns
off the MRI branches, the mirroring, the target-size fit and the
nearest-neighbour interpolation, leaving a generic "put the cohort on one voxel
grid and recentre it". So it is vendored into VFACE's own `resample.py`, which
is what CONTRIBUTING.md says two tools needing the same code usually get.

The cost is worth writing down here rather than only in the tool's README:
**resampling interpolates, so it changes voxel values.** A VFACE result depends
on that copy and not on MRI2CBCT, and if MRI2CBCT is ever ported the two can
drift apart without anything failing.

**And one upstream oddity is reproduced rather than repaired.**
`Measure.__SignMeaningDist` resolves a midpoint's side and then overwrites its
own answer, so `Mid_ROr_LOr` reads as `M` rather than `R` or `L`. That label
decides the sign of a feature the asymmetry models were trained on, so
correcting it would change classifications on models already trained -- a
clinical decision rather than a port's. It is in `measure._side_of`, with a test.

`tools/_template/` has no row: it is the reference package the others are copied
from, not a port.

**Every tool is in this table.** AREG used to be the one this document said was
deliberately absent; it was migrated on 2026-08-14, and VFACE followed on
2026-09-21.

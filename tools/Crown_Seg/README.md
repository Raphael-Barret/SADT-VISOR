# sadt-crownseg

Labels every tooth of an intraoral surface scan with its dental number, as a
point-data array on a copy of the mesh. That array is the precondition for ALI's
IOS landmark identification and for the IOS modes of ASO, AREG and FlexReg,
which is why this is its own tool.

## Provenance

The method, the camera rig, the network and the weights are shapeaxi's
`dentalmodelseg` (DCBIA-OrthoLab, <https://github.com/DCBIA-OrthoLab/ShapeAXI>),
which the Slicer modules ran as an executable. This tool no longer calls
shapeaxi: it imports pytorch3d, which has no Windows wheel and no Metal kernel,
and its import chain pulls in lightning, ocnn, h5py, scikit-learn, torchvision,
nibabel and itk for two small classes. `engine.py` carries the same pipeline on
**torch, monai, VTK and Embree** and nothing else.

| Part | Source |
|---|---|
| surface reading, unit scaling, normals, island removal, closing, FDI table | shapeaxi's own functions, extracted by AST into `_shapeaxi.py` (two marked edits) |
| 42 cameras, 320x320, 60 degree field of view, white background, normals plus depth, per-face probability sum, per-vertex write | shapeaxi's `DentalModelSeg` and `prediction`, re-expressed in `engine.py` |
| ray casting and shading | `render_native.py`, the renderer ALI_IOS and AREG_IOS already ship (a byte-identical copy, kept so by `scripts/tests/test_vendored_copies.py`) |
| the idea of vendoring the pipeline, and the white-background finding | the lab's `iosseg` package, whose VTK and Open3D renderer is Eduardo Caleme's |

Carried over from `slicer-remote-tool-server`'s `tools/CrownSeg/`, whose history
this repository holds; `git log --follow` on `src/sadt_crownseg/pipeline.py`
reaches back through it.

## What it does

| | |
|---|---|
| Inputs | `meshes`: one `.vtk`/`.stl` surface or a folder of them, searched recursively. `model`: the checkpoint. `output_dir`: where results go. |
| Outputs | One labelled `.vtk` per mesh, named `<stem>_<suffix>.vtk`, plus `run_report.json`. The input tree is mirrored under `output_dir`; the meshes the network actually ran on are filed one level down, under `crownseg_input_<suffix>/`, the layout shapeaxi's csv branch used. `run_report.json` names every output, so a caller never has to know which branch a mesh took. |
| Model files | One checkpoint, `07-21-22_val-loss0.169.pth` (6.6 MB), from the Fly-by-CNN 3.0 release. It stays in `DATA/Crown_Seg/models/`; it is never part of this repository. |
| Devices | `cuda`, `cpu`, and `mps` on macOS. The network runs on the device; the ray casting and the vote run on the host. |

The published contract (arguments, types, defaults, choices, output names and
the report) is the one the shapeaxi-based tool had. `num_workers` is still
accepted and no longer does anything: it sized shapeaxi's DataLoader.

- **An already-labelled mesh is passed through, not re-predicted.** Pass
  `skip_segmented=False` to force the network to run anyway.
- **A pre-segmented `.stl` is impossible** -- STL carries no point data by
  construction -- so every output is `.vtk`, whichever branch a mesh took.
- **`numbering` changes the integers written into the array**, not the mesh.
  Whatever consumes the result has to agree; the report records which was used.

## The checkpoint

Read once and hashed, and the same bytes are given to `torch.load(...,
weights_only=True, map_location="cpu")`: a file that holds anything but tensors
and plain containers is refused, not executed. A file named
`07-21-22_val-loss0.169.pth` must have sha256
`9fa232d2eca9bf7bf6b6e7b03d48828d14f9369448e5ea7bfeb574474daabdbe`, or the run
is refused (`ToolInputError`, answered 422). Another vintage under another name
is loaded and its digest is logged, since `find_checkpoint` exists to allow it.
The state dict is loaded strictly and no key prefix is stripped: the keys begin
with `model.` because that is monai's own container name, and stripping it
matches nothing.

## Determinism

shapeaxi's result depends on luck in two places, and both are settled here:

1. The cameras that sit on an axis (8 of 42, by shapeaxi's own test) are nudged
   by `torch.normal` drawn from the global generator, so a view can be oriented
   differently on each run. The nudge now comes from a generator seeded with
   `CAMERA_SEED`.
2. `P_faces[:, pf] += pred` and `V_labels[faces_pid0] = P_faces` write to
   repeated indices, and torch leaves the winner undefined: arbitrary on CUDA,
   and racy on the CPU too once the thread pool is used (the same line gave a
   different array on each call with 28 threads, and matched the last write only
   with one). Both writes are done in numpy on the host, where the last value is
   documented to win, after the network.

This keeps shapeaxi's labels (last pixel of a face, last face of a vertex) and
does **not** claim they are the right vote: one pixel per face per view and one
face per vertex are kept and the rest discarded. `engine.predict(...,
accumulate_pixels=True)` sums every pixel instead, so the alternative can be
measured; it is not the default because it moves the labels away from the
published method.

## Versions

torch 2.11.0+cu128, vtk 9.6.2, numpy 2.3.2, Python 3.11; the `segmentation`
extra adds monai 1.6.0 and embreex 4.4.0.post1.

`uv lock` holds 49 packages where the shapeaxi stack held 172. Nothing links
against torch's ABI any more, so torch is pinned for reproducibility only, and
there is no pytorch3d wheel index, no torchvision and no CUDA toolkit to match.
Python stays at 3.11, the deployment image's interpreter, so the lock and the
image do not move with this change. open3d is not a dependency.

## Working on it

```bash
cd tools/Crown_Seg
uv sync                          # torch, vtk, numpy
uv run pytest                    # the engine is stubbed; engine tests skip without the extra
uv sync --extra segmentation     # monai and embreex, two small packages
uv run pytest                    # now the engine tests run as well, on a synthetic mesh
SADT_CROWNSEG_MODEL=... SADT_CROWNSEG_MESH=... uv run pytest -m models -o addopts=
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```

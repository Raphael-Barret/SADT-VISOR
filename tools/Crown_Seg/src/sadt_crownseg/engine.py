"""The crown-segmentation engine: shapeaxi's `dentalmodelseg`, without shapeaxi.

What this module is, in one paragraph. shapeaxi's `DentalModelSeg` renders an
arch from 42 cameras, runs a 2D U-Net on every view, adds the class
probabilities of every pixel to the triangle it came from, takes the argmax per
triangle, hands the label to each triangle's first vertex and cleans the result
up (islands, then a morphological closing). This module is that, call for call,
with two parts replaced and nothing else:

* pytorch3d's rasterizer and shader are replaced by `render_native`, the
  renderer ALI_IOS and AREG_IOS already ship (Embree for the rays, torch for the
  rest), which was measured against pytorch3d on CUDA;
* lightning, `TimeDistributed`'s package and the dataset/dataloader are gone:
  the network is a plain monai U-Net called on the 42 views as one batch, which
  is the same computation.

The surface helpers (`_shapeaxi.py`) and the post-processing are shapeaxi's own
functions, extracted by AST. The method, the camera rig and the weights are the
DCBIA-OrthoLab group's (https://github.com/DCBIA-OrthoLab/ShapeAXI). The idea of
vendoring the pipeline without pytorch3d, and the white background finding
below, come from the lab's `iosseg` package, whose renderer is Eduardo Caleme's.

THE ONE SETTING THAT IS EASY TO GET WRONG. shapeaxi builds its shader without
`BlendParams`, so pytorch3d paints the background WHITE. Rendering the arch on
black instead costs more than half the tooth coverage, because the U-Net has a
five-level encoder and the surround changes what it predicts on the arch itself,
worst at the silhouette. `BACKGROUND` is white for that reason and is not a
free parameter.

DETERMINISM. Two things in shapeaxi make a run depend on luck, and both are
settled here rather than inherited:

* the cameras that sit exactly on an axis are nudged by `torch.normal` drawn
  from the GLOBAL generator, so the pole views come out differently on each run.
  The nudge comes from a generator seeded with `CAMERA_SEED` instead;
* `V_labels_prediction[faces_pid0] = P_faces` (and, within a view,
  `P_faces[:, pf] += pred`) write to repeated indices. torch leaves the winner
  undefined: it is the last write on the CPU and an arbitrary one on CUDA, which
  is what made the same code disagree with itself across devices. Both writes
  are done here in numpy, where "the last value is assigned" is documented
  behaviour, on the host, after the network. Every device therefore resolves
  them identically, and the result is the one shapeaxi gives on the CPU.

That keeps shapeaxi's results and is NOT a claim that last-write-wins is the
right vote: it picks one pixel of a face and one face of a vertex and discards
the rest. `predict(..., accumulate_pixels=True)` adds up every pixel instead, so
the alternative can be measured; it is not the default because it moves the
labels away from the published method.
"""

import contextlib
import hashlib
import io
import logging
import os

import monai
import numpy as np
import torch
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from . import _shapeaxi as sx
from . import render_native
from .errors import ToolInputError

logger = logging.getLogger(__name__)

# The network and its input, as shapeaxi's DentalModelSeg builds them.
OUT_CHANNELS = 34
GINGIVA_LABEL = 33
IMAGE_SIZE = 320
RADIUS = 1.35
SUBDIVISION_LEVEL = 2
# pytorch3d's FoVPerspectiveCameras() defaults, which is what shapeaxi uses.
FOV = 60.0
ZNEAR = 1.0
# HardPhongShader with no BlendParams: pytorch3d's default is white.
BACKGROUND = (1.0, 1.0, 1.0)

# Seeds the nudge of the cameras that sit on an axis; see the module docstring.
CAMERA_SEED = 0

# The published checkpoint, and its digest. A file carrying the published NAME
# must carry these bytes: the name is what `pipeline.find_checkpoint` keys on and
# what the run report records, so a file that says it is the published weights
# and is not is refused rather than run. Another vintage under another name is
# loaded, and its digest is logged.
PUBLISHED_CHECKPOINT = "07-21-22_val-loss0.169.pth"
PUBLISHED_SHA256 = "9fa232d2eca9bf7bf6b6e7b03d48828d14f9369448e5ea7bfeb574474daabdbe"

# The FDI / Universal lookup, shapeaxi's ConvertFDI.
FDI_LUT = np.array([0, 18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27,
                    28, 38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47,
                    48, 0])


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

def create_network():
    """shapeaxi's `create_unet_model`, unchanged."""
    return monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=OUT_CHANNELS,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def load_network(checkpoint_path):
    """The U-Net with the checkpoint's weights, or a refusal that says why.

    The file is read ONCE, hashed, and the same bytes are handed to
    `torch.load`, so what was checked is what was loaded. `weights_only=True`
    means a checkpoint is refused if it holds anything but tensors and plain
    containers: a trapped file is not executed. `map_location="cpu"` because the
    published file was saved from a CUDA device and a machine without CUDA could
    not otherwise read it.

    The state dict is loaded STRICTLY and no key prefix is stripped. The keys
    begin with `model.` because that is monai's own container name; stripping
    it matches no parameter, and `strict=False` would then load nothing and let
    the random network answer everything at about 0.15 confidence.
    """
    checkpoint_path = os.fspath(checkpoint_path)
    try:
        with open(checkpoint_path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ToolInputError(
            f"The checkpoint '{os.path.basename(checkpoint_path)}' cannot be read: {exc.strerror}."
        ) from exc

    digest = hashlib.sha256(data).hexdigest()
    name = os.path.basename(checkpoint_path)
    if name == PUBLISHED_CHECKPOINT and digest != PUBLISHED_SHA256:
        raise ToolInputError(
            f"'{name}' is not the published crown-segmentation checkpoint: its sha256 "
            f"is {digest}, the published one is {PUBLISHED_SHA256}. Fetch it again "
            "with `setup-models.sh`, or give the file another name if it is "
            "a different vintage."
        )
    logger.info("checkpoint %s, sha256 %s", name, digest)

    try:
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:  # pickle.UnpicklingError, RuntimeError, EOFError ...
        raise ToolInputError(
            f"'{name}' is not a plain PyTorch state dict (torch.load with "
            f"weights_only=True refused it: {type(exc).__name__})."
        ) from exc

    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    network = create_network()
    try:
        network.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ToolInputError(
            f"'{name}' does not match the crown-segmentation network "
            f"({type(exc).__name__}). It must be a state dict of the monai U-Net "
            "(4 inputs, 34 classes) whose keys begin with `model.`."
        ) from exc
    return network


# ---------------------------------------------------------------------------
# The camera rig
# ---------------------------------------------------------------------------

def camera_rig(seed=CAMERA_SEED):
    """The 42 camera positions, (42, 3) float32: a twice-subdivided icosahedron.

    A camera sitting exactly on an axis makes `look_at_rotation` singular, so
    shapeaxi nudges it by `normal(0, 1e-7)`. The nudge decides how that
    degenerate view resolves its up vector, which is why it must not be left to
    the global generator; see the module docstring.
    """
    rig, _faces = sx.PolyDataToTensors_v_f(sx.CreateIcosahedronSubdivided(RADIUS, SUBDIVISION_LEVEL))
    rig = rig.to(torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    for index, vertex in enumerate(rig):
        if torch.abs(torch.sum(vertex)) == RADIUS:
            rig[index] = vertex + torch.normal(0.0, 1e-7, (3,), generator=generator)
    return rig


def camera_pose(position):
    """`(R (1, 3, 3), T (1, 3))` for a camera at `position` (1, 3) looking at the origin."""
    rotation = render_native.look_at_rotation(position)
    translation = -torch.bmm(rotation.transpose(1, 2), position[:, :, None])[:, :, 0]
    return rotation, translation


def ray_rotation(rotation):
    """The rotation `render_native` must be given so that it casts pytorch3d's rays.

    pytorch3d turns a view-space ray into a world ray with the INVERSE of `R`;
    `render_native` uses its transpose, which is the same thing for an
    orthonormal matrix. One camera of the rig is the exception: it sits so near
    the `up` axis that `look_at_rotation` returns a matrix with an axis of norm
    0.02, and the transpose then renders a full view where pytorch3d renders a
    strip a few pixels wide (measured: 36 000 pixels differ on that one view).
    The strip is what shapeaxi's network sees, so it is reproduced: for a
    non-orthonormal matrix the transpose of the inverse is passed instead, and an
    orthonormal one is returned untouched, so the other cameras do not move.
    """
    wide = rotation.double()
    if torch.allclose(wide @ wide.transpose(1, 2), torch.eye(3, dtype=torch.float64), atol=1e-4):
        return rotation
    return torch.linalg.inv(wide).transpose(1, 2).to(rotation.dtype)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_surface(path):
    """Read a scan and prepare the network's inputs.

    What shapeaxi's `SaxiDataset.__getitem__` does with `UnitSurfTransform`:
    scale into a unit sphere, then colour every vertex by its surface normal
    (that is the image the network reads; there is no lighting). Returns the
    ORIGINAL surface, which is what gets labelled and written, and the tensors
    `(verts (V, 3), faces (F, 3), colors (V, 3))` of the scaled copy.
    """
    surf = sx.ReadSurf(path)
    if surf.GetNumberOfPoints() == 0:
        raise ToolInputError("No points were read from the mesh. Is it a valid surface file?")
    unit = sx.GetUnitSurf(surf)
    verts, faces = sx.PolyDataToTensors_v_f(unit)
    normals = sx.ComputeNormals(unit)
    colors = torch.tensor(vtk_to_numpy(sx.GetColorArray(normals, "Normals"))).to(torch.float32) / 255.0
    return surf, verts, faces, colors


# ---------------------------------------------------------------------------
# Rendering and voting
# ---------------------------------------------------------------------------

def render_views(verts, faces, colors, rig, image_size=None):
    """Every camera's view: `(X (n, 4, S, S), pix_to_face (n, S, S))`, on the CPU.

    The four channels are the interpolated normal colour and the view-space
    depth, as shapeaxi stacks them (`images[..., 0:3]` and `zbuf`).
    """
    image_size = IMAGE_SIZE if image_size is None else image_size
    mesh = render_native.Mesh(verts, faces, colors)
    renderer = render_native.Renderer(
        render_native.AmbientLights(), fov=FOV, znear=ZNEAR, size=image_size, background=BACKGROUND
    )
    images, hits = [], []
    for position in rig:
        rotation, translation = camera_pose(position.unsqueeze(0))
        rgba, fragments = renderer(meshes_world=mesh, R=ray_rotation(rotation), T=translation)
        images.append(torch.cat([rgba[0, :, :, 0:3], fragments.zbuf[0]], dim=-1).permute(2, 0, 1))
        hits.append(fragments.pix_to_face[0, :, :, 0])
    return torch.stack(images), torch.stack(hits)


def accumulate_view(totals, pix_to_face, probabilities, accumulate_pixels=False):
    """Add one view's probabilities to the per-face totals `(classes, faces)`, in place.

    Default: shapeaxi's `P_faces[:, pf] += pred`, whose repeated indices keep
    the LAST pixel's value (see the module docstring); a background pixel
    (`pf == -1`) indexes the last face, exactly as there. `accumulate_pixels`
    sums every pixel that hit a face instead and ignores the background.
    """
    index = pix_to_face.reshape(-1)
    values = probabilities.reshape(probabilities.shape[0], -1)
    if accumulate_pixels:
        hit = index >= 0
        for cls in range(totals.shape[0]):
            totals[cls] += np.bincount(index[hit], weights=values[cls][hit], minlength=totals.shape[1]).astype(
                totals.dtype
            )
        return
    totals[:, index] = totals[:, index] + values


def vertex_labels(faces, face_labels, vertex_count):
    """Each face's label carried to its first vertex, `V[faces[:, 0]] = labels`.

    A vertex that is the first corner of several faces keeps the label of the
    last of them (numpy documents that the last value is assigned); a vertex
    that is nobody's first corner stays 0.
    """
    labels = np.zeros(vertex_count, dtype=np.int64)
    labels[np.asarray(faces)[:, 0]] = face_labels
    return labels


def predict(network, rig, verts, faces, colors, device="cpu", image_size=None, accumulate_pixels=False):
    """Per-vertex labels `(V,) int64` for one scaled surface, before post-processing.

    shapeaxi's `prediction()`: the views go through the network as one batch,
    the softmax is taken over the classes of the pixels that hit a face (a miss
    has its logits zeroed first, as there), probabilities are added per face
    over all views and the argmax is taken once, at the end. A per-pixel argmax
    vote would collapse tooth coverage to about 7%. Everything after the network
    runs on the host.
    """
    views, pix_to_face = render_views(verts, faces, colors, rig, image_size)
    covered = (pix_to_face >= 0)[:, None]

    with torch.no_grad(), deterministic_kernels(device):
        logits = network(views.to(device))
        probabilities = torch.softmax(logits * covered.to(device, dtype=logits.dtype), dim=1)
        probabilities = probabilities.cpu().numpy()
    del logits, views

    totals = np.zeros((OUT_CHANNELS, faces.shape[0]), dtype=np.float32)
    hits = pix_to_face.numpy()
    for view in range(hits.shape[0]):
        accumulate_view(totals, hits[view], probabilities[view], accumulate_pixels)
    return vertex_labels(faces.numpy(), np.argmax(totals, axis=0), verts.shape[0])


@contextlib.contextmanager
def deterministic_kernels(device):
    """Pin the cuDNN choices that could change a result between two runs.

    Process flags, not environment variables, put back on the way out. They
    matter on CUDA only: the CPU kernels are fixed for a given thread count and
    MPS has no such switch.
    """
    if not str(device).startswith("cuda"):
        yield
        return
    before = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before


# ---------------------------------------------------------------------------
# Post-processing and output
# ---------------------------------------------------------------------------

def post_process(surf, labels, out_channels=OUT_CHANNELS):
    """shapeaxi's `post_processing`: gum islands, every island, then a closing per label.

    Modifies the vtk array `labels` in place. The raw network output is meant to
    be speckled; this is a designed step, and skipping it looks exactly like a
    broken model.
    """
    sx.RemoveIslands(surf, labels, GINGIVA_LABEL, 500, ignore_neg1=True)
    for label in range(out_channels):
        sx.RemoveIslands(surf, labels, label, 200, ignore_neg1=True)
    for label in range(out_channels):
        sx.DilateLabel(surf, labels, label, iterations=2, dilateOverTarget=False, target=None)
        sx.ErodeLabel(surf, labels, label, iterations=2, target=None)


def to_fdi(surf, array_name):
    """Renumber the labels Universal to FDI under the same array name, shapeaxi's `ConvertFDI`."""
    labels = FDI_LUT[vtk_to_numpy(surf.GetPointData().GetScalars(array_name))]
    converted = numpy_to_vtk(labels)
    converted.SetName(array_name)
    surf.GetPointData().AddArray(converted)
    return surf


def segment_mesh(network, rig, mesh_path, output_path, array_name, fdi=False, device="cpu"):
    """Label one scan and write it. The input is read once and never modified."""
    surf, verts, faces, colors = load_surface(mesh_path)
    labels = predict(network, rig, verts, faces, colors, device=device)

    labels_vtk = numpy_to_vtk(labels)
    labels_vtk.SetName(array_name)
    surf.GetPointData().AddArray(labels_vtk)
    post_process(surf, labels_vtk)
    if fdi:
        surf = to_fdi(surf, array_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sx.Write(surf, output_path, print_out=False)
    return output_path

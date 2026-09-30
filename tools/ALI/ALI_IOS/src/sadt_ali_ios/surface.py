"""Reading an intraoral mesh and turning it into the tensors the network wants.

Ported from ALI_IOS_utils/surface.py. The scaling, the normals-as-colors
encoding and the label array names are all part of what the shipped weights
were trained against, so they are reproduced exactly.
"""

import logging
import os

import numpy as np

from .errors import ToolUnavailableError

logger = logging.getLogger(__name__)

# Point-data arrays that carry per-tooth labels, in the order they are tried.
# The same three names CrownSeg writes and recognizes.
LABEL_ARRAY_NAMES = ("PredictedID", "UniversalID", "Universal_ID")

_VTK_HINT = "ALI's IOS engine needs VTK. Run `uv sync` in tools/ALI."


def import_vtk():
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_VTK_HINT} (missing: vtk)") from exc
    return vtk, vtk_to_numpy


def read_surface(path: str):
    """Read a .vtk or .stl mesh.

    Only the two extensions the schema advertises. VTK reads more (.vtp,
    .obj, .off) and the original module's reader listed them, but discovery
    never reached them -- an accepted-then-ignored format is exactly the trap
    `.stl` fell into, and one list is better than two that disagree.
    """
    vtk, _ = import_vtk()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Surface file not found: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
    else:
        raise ValueError(f"Unsupported surface format '{extension}'. Expected .vtk or .stl.")

    reader.SetFileName(path)
    reader.Update()
    surface = reader.GetOutput()
    if surface.GetNumberOfPoints() == 0:
        raise ValueError("Surface has no points")
    return surface


def label_array_name(surface) -> str:
    """Which tooth-label array this mesh carries, or None if it carries none."""
    point_data = surface.GetPointData()
    present = {point_data.GetArrayName(index) for index in range(point_data.GetNumberOfArrays())}
    for name in LABEL_ARRAY_NAMES:
        if name in present:
            return name
    return None


def scale_to_unit(surface):
    """Centre the mesh on its bounding box and scale it into the unit sphere.

    Returns (scaled copy, centre, scale factor); the last two are what
    `upscale` needs to put a predicted point back in the patient's own
    millimetres.
    """
    vtk, _ = import_vtk()

    copy = vtk.vtkPolyData()
    copy.DeepCopy(surface)
    points = copy.GetPoints()

    bounds = points.GetBounds()
    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]
    )
    bounds_max = np.array([max(bounds[0], bounds[1]), max(bounds[2], bounds[3]),
                           max(bounds[4], bounds[5])])

    coordinates = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    scale_factor = 1 / np.linalg.norm(bounds_max - center)
    scaled = (coordinates - center) * scale_factor

    for index, point in enumerate(scaled):
        points.SetPoint(index, point)
    copy.SetPoints(points)
    return copy, center, scale_factor


def upscale(position, center, scale_factor):
    """A point in unit-sphere space, back in the patient's own coordinates."""
    return (np.asarray(position) / scale_factor) + center


def _compute_normals(surface):
    vtk, _ = import_vtk()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def _normals_as_colors(surface, vtk_to_numpy):
    """Per-point normals mapped into [0, 1] RGB, the network's input texture."""
    normals = surface.GetPointData().GetArray("Normals")
    if normals is None:
        raise ValueError("Surface has no 'Normals' array after normal computation")
    array = vtk_to_numpy(normals)
    return (array * 0.5 + 0.5).astype(np.float32)


def surface_properties(scaled_surface, device):
    """(vertices, faces, colors, tooth labels) as batched tensors.

    A mesh with no label array raises instead of falling back to zeros the way
    the original did: zeros mean "every vertex is tooth 0", so no tooth is ever
    found and the run ends reporting no landmarks with no reason given. The
    engine checks every mesh for labels up front, so this is the backstop.
    """
    from .torch_helpers import import_torch

    torch = import_torch()
    _, vtk_to_numpy = import_vtk()

    with_normals = _compute_normals(scaled_surface)

    colors = torch.tensor(_normals_as_colors(with_normals, vtk_to_numpy), dtype=torch.float32,
                          device=device)
    vertices = torch.tensor(
        vtk_to_numpy(with_normals.GetPoints().GetData()), dtype=torch.float32, device=device
    )
    # VTK stores polys as (count, id0, id1, id2) tuples; the meshes here are
    # triangulated, so dropping the leading count gives the face table.
    faces = torch.tensor(
        vtk_to_numpy(with_normals.GetPolys().GetData()).reshape(-1, 4)[:, 1:],
        dtype=torch.int64,
        device=device,
    )

    name = label_array_name(with_normals)
    if name is None:
        raise ValueError(
            "This mesh carries no tooth labels. Expected a point-data array named one of: "
            + ", ".join(LABEL_ARRAY_NAMES)
        )
    labels = torch.tensor(
        vtk_to_numpy(with_normals.GetPointData().GetScalars(name)).astype(np.int64),
        dtype=torch.int64,
        device=device,
    )
    labels = torch.clamp(labels, min=0)

    return (
        vertices.unsqueeze(0),
        faces.unsqueeze(0),
        colors.unsqueeze(0),
        labels.unsqueeze(0),
    )


def faces_on_tooth(faces, face_ids, labels, tooth_number: int) -> list:
    """Keep only the faces that actually belong to this tooth.

    The network predicts on a rendered view, so a mask can spill onto the
    neighbouring tooth or the gum; a face is kept when at least one of its
    vertices carries this tooth's label.

    **One kernel, not three per face.** This read `int(label_table[vertex])`
    inside a nested Python loop over every predicted face and each of its three
    corners. Both tables live on the card, so each of those reads was a device
    synchronisation: measured on the reference mesh, 2.09 s of a 21.6 s run
    (9.7 %) to answer a question about 119,456 faces that one indexing
    operation answers at once.

    The filter is the same filter, evaluated in the same order over the same
    input, so the list returned is identical element for element -- verified
    against the loop on all 70 calls of a real run. Order matters beyond
    tidiness: the caller sums these faces' vertices sequentially, and float32
    addition is not associative.
    """
    from .torch_helpers import import_torch

    if not len(face_ids):
        return []

    torch = import_torch()
    face_table = faces.squeeze(0)
    label_table = labels.squeeze(0)
    wanted = torch.as_tensor(face_ids, dtype=torch.long, device=face_table.device)
    # (faces, 3) labels -> one bool per face, True when any corner is this tooth.
    keep = (label_table[face_table[wanted]] == int(tooth_number)).any(dim=1)
    return wanted[keep].tolist()

"""Reading, writing and transforming intra-oral surface meshes, plus the jaw a
file name claims to be.

Ported from `AREG_IOS/AREG_IOS_utils/utils.py` (ReadSurf/WriteSurf/
VTKMatrixToNumpy), `transformation.py` (TransformSurf) and the jaw half of
`dataset.py` (isLowerUpper/removeLowerUpper).

Deliberately AREG's own copy rather than an import of `tools/ASO/src/ios/
surfaces.py`: importing another tool's module at load time makes one tool's
missing dependency take both out of the registry. The two are close but not
identical -- this one has the jaw vocabulary AREG needs and no .off reader,
AREG's IOS engine never seeing a .off.

Two behaviours differ from the original:

* meshes are written BINARY. `vtkPolyDataWriter` defaults to ASCII, and the
  difference is not cosmetic: binary is smaller, ~100x faster to parse, and the
  MORE accurate of the two, round-tripping float32 exactly where ASCII prints
  six significant digits and moves points on read-back;
* a mesh whose name does not say its jaw is REFUSED rather than treated as a
  lower arch. `isLowerUpper(file, "Upper")` returning False was taken to mean
  "lower", so a maxillary mesh named `patient1.vtk` was registered against the
  mandibular timepoint and returned as a success.
"""

import os
import re

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from sadt_areg_common import catalogs, pairing

SURFACE_EXTENSIONS = (".vtk", ".vtp", ".stl", ".obj")

# The point arrays a crown-segmented intra-oral scan may carry, most specific
# first. Slicer's tools have used all three over time.
LABEL_ARRAY_NAMES = ("Universal_ID", "PredictedID", "UniversalID")

_SEPARATORS = re.compile(r"[_\-.\s]+")


class SurfaceError(Exception):
    """A mesh could not be read, or does not carry what the mode needs."""


def is_surface_file(filename: str) -> bool:
    return filename.lower().endswith(SURFACE_EXTENSIONS)


def read_surface(path: str) -> vtk.vtkPolyData:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllFieldsOn()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
    else:
        raise SurfaceError(
            f"'{os.path.basename(path)}': unsupported surface format. Expected one of "
            f"{', '.join(SURFACE_EXTENSIONS)}."
        )

    reader.SetFileName(path)
    reader.Update()
    surface = reader.GetOutput()
    if surface is None or surface.GetNumberOfPoints() == 0:
        raise SurfaceError(f"'{os.path.basename(path)}' holds no geometry.")
    return surface


def write_surface(surface: vtk.vtkPolyData, path: str) -> str:
    """Write a mesh, keeping its per-point arrays. Returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".vtp"):
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()
    writer.SetFileName(path)
    writer.SetInputData(surface)
    writer.Update()
    return path


def output_extension(input_path: str) -> str:
    """The format a result mesh is written in.

    `.vtp` in gives `.vtp` out; everything else becomes `.vtk`. The tooth-label
    array is what makes an intra-oral result usable downstream and `.stl` cannot
    hold it, so `.stl` is read but never written back.
    """
    return ".vtp" if input_path.lower().endswith(".vtp") else ".vtk"


def label_array_name(surface: vtk.vtkPolyData) -> str:
    """The per-point tooth-label array this mesh carries, or None."""
    point_data = surface.GetPointData()
    present = {point_data.GetArrayName(index) for index in range(point_data.GetNumberOfArrays())}
    for name in LABEL_ARRAY_NAMES:
        if name in present:
            return name
    return None


def all_meshes_carry_labels(root: str) -> bool:
    """True when every mesh under `root` already carries its tooth labels.

    Which is the question `automation` used to put to the clinician: a
    Semi-Automated IOS registration takes crown-segmented meshes, a
    Fully-Automated one labels them with Crown_Seg first. The answer is written
    in the file, in one of `LABEL_ARRAY_NAMES` -- so asking meant the answer
    could disagree with the folder, and when it did a run either relabelled
    meshes that were already labelled or failed on meshes that were not.

    EVERY mesh, and short-circuiting on the first that has none: labelling is
    done for the whole folder or not at all, so one unlabelled mesh in forty
    makes the cohort one to label. `False` is also the answer for a folder with
    no mesh in it -- there is nothing here that is ready, and the caller gets
    the refusal that names the real problem a step later.
    """
    found = False
    for directory, _subdirs, names in os.walk(root or ""):
        for name in sorted(names):
            if name.startswith(".") or not is_surface_file(name):
                continue
            found = True
            try:
                if label_array_name(read_surface(os.path.join(directory, name))) is None:
                    return False
            except SurfaceError:
                # A mesh that cannot be read is not a mesh that is ready.
                return False
    return found


def jaw_of(filename: str) -> str:
    """'Upper', 'Lower', or None when the name does not say.

    Matched on whole tokens of the stem. The original used substrings, and its
    Upper vocabulary included the bare `_U` and `U_` -- so a patient identifier
    like `P_U12` was an upper arch whatever the file actually held, and a
    subject folder named `Mdx` made every mesh in it a mandible.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    # `pairing.tokens` rather than a local split: it also separates a jaw run
    # together with a timepoint, `UpperT1` -> `upper` + `t1`, which upstream's
    # own test set is named with. A local split saw one token, `uppert1`, and
    # reported no jaw at all.
    for token in pairing.tokens(stem):
        jaw = catalogs.JAW_TOKENS.get(token)
        if jaw:
            return jaw
    return None


def transform_surface(surface: vtk.vtkPolyData, matrix: np.ndarray) -> vtk.vtkPolyData:
    """Apply a 4x4 matrix to a mesh, returning a new one."""
    copy = vtk.vtkPolyData()
    copy.DeepCopy(surface)

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.asarray(matrix, dtype=np.float64).reshape(16).tolist())

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(copy)
    transform_filter.Update()
    return transform_filter.GetOutput()


def points_of(surface: vtk.vtkPolyData) -> np.ndarray:
    return vtk_to_numpy(surface.GetPoints().GetData())


def matrix_from_vtk(matrix) -> np.ndarray:
    return np.array(
        [[matrix.GetElement(row, column) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def compute_normals(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def scale_to_unit(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Centre a mesh on its bounding box and scale it into the unit sphere.

    What the network was trained on. Rewritten in numpy: the original walked
    every vertex twice through `GetPoint`/`SetPoint`, which on a 200k-vertex
    intra-oral scan is the slowest step of the whole preprocessing.
    """
    copy = vtk.vtkPolyData()
    copy.DeepCopy(surface)
    points = copy.GetPoints()

    bounds = np.asarray(points.GetBounds(), dtype=np.float64)
    center = np.array(
        [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
    )
    extreme = np.array([max(bounds[0], bounds[1]), max(bounds[2], bounds[3]), max(bounds[4], bounds[5])])
    span = np.linalg.norm(extreme - center)
    if span == 0:
        raise SurfaceError("mesh has zero extent, nothing to scale")

    scaled = (vtk_to_numpy(points.GetData()) - center) / span
    from vtk.util.numpy_support import numpy_to_vtk

    points.SetData(numpy_to_vtk(np.ascontiguousarray(scaled, dtype=np.float32), deep=True))
    copy.SetPoints(points)
    return copy

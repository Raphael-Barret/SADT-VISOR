"""Surface (.vtk) generation from a segmentation, for AMASSS's
"generate surface file" option.

Surfaces are generated here rather than in the Slicer client so that every
consumer gets them, not only the ones running inside Slicer.

The mesh pipeline produces the SAME GEOMETRY as the original CLI's, because
these surfaces are already consumed downstream and changing them would be a
regression. Two stages of it were replaced once each replacement was shown to
be exact -- same points, same cells, same triangles -- on a real mandible of
589 934 triangles, and not merely close:

  * the array reaches VTK DIRECTLY instead of through a temporary `.nrrd`
    written to disk and read back. 96 MB written and reread per mask, 18 masks
    per scan on a full run: 3.4 GB of round trip to move a buffer that was
    already in memory. Measured 1.28 s -> 0.63 s for the input plus contour.
  * `vtkFlyingEdges3D` at 0.5 replaces `vtkDiscreteMarchingCubes`. On a binary
    0/1 mask the 0.5 isosurface passes exactly through the midpoint of every
    crossed edge, which is where the discrete filter puts its vertices too --
    hence the identity. Measured 0.55 s -> 0.14 s.

What was NOT changed, having been measured and refused: `vtkQuadricClustering`
is forty times faster than `vtkDecimatePro` and moves every point by 0.065 mm
on average; decimating before smoothing saves 0.57 s and gives a different
surface; contouring on a grid coarsened by 1.5 is three times faster and moves
the surface by 0.120 mm. All three are visible changes to a clinical output,
and the brief for this work was that the result does not move.

What is fixed is the crash: the original resolved a structure's label by
parsing it out of the output FILE NAME and looking it up in `LABELS["LARGE"]`,
a KeyError for any structure absent from that table which aborted the whole
scan. The structure code is passed in explicitly now.

vtk, numpy and SimpleITK are imported inside the functions that use them. They
are hard dependencies of this package now -- the server-side port had to cope
with them being absent, and `is_available()` plus its ToolUnavailableError
guard went with that -- but schema generation still imports this module and
must not pay for them.
"""

import logging

logger = logging.getLogger(__name__)


def _image_from_mask(mask, reference):
    """The mask as a `vtkImageData`, without touching the disk.

    numpy's last axis is contiguous and VTK's first is, which is why the
    dimensions are handed over reversed: the buffer is shared in the order it
    already has rather than transposed into a copy.
    """
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    image = vtk.vtkImageData()
    depth, height, width = mask.shape
    image.SetDimensions(width, height, depth)
    image.SetSpacing(*reference.GetSpacing())
    image.SetOrigin(*reference.GetOrigin())
    flat = np.ascontiguousarray(mask.astype(np.uint8)).ravel(order="C")
    # deep=True: the numpy buffer is a temporary of the caller's, and VTK must
    # not be left pointing into it once this returns.
    scalars = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    scalars.SetName("mask")
    image.GetPointData().SetScalars(scalars)
    return image


def _mesh_from_mask(mask, reference, smoothing: int, color_rgb,
                    decimation: int = 0):
    """Build a colored surface for one binary mask.

    `decimation` is the percentage of triangles to drop (0 keeps the raw
    marching-cubes mesh). See the `surface_decimation` argument in AMASSS.py
    for why the default is not 0.
    """
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    image = _image_from_mask(mask, reference)

    # 0.5 on a 0/1 mask: the isosurface crosses every edge at its midpoint,
    # which is exactly where the discrete filter placed its vertices. Verified
    # equal -- points, cells and triangles -- on a real mandible.
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.5)

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(max(0, int(smoothing)))
    smoother.Update()

    polydata = smoother.GetOutput()

    # Marching cubes works on the ORIGINAL scan grid, so a CBCT at 0.33mm
    # yields a triangle per voxel face: 1.6M for a cranial base, 11.8M for
    # a merged nine-structure volume. That is a level of detail no CBCT
    # segmentation actually carries -- a binary mask is only accurate to
    # about half a voxel to begin with -- and it is what made the results
    # unusable downstream, both to ship and to open.
    #
    # Decimating 90% of a cranial base moves the surface by 0.059mm on
    # average (p95 0.171mm), a fifth of a voxel, well inside the mask's own
    # uncertainty. PreserveTopologyOn keeps thin structures from being
    # punctured; the reduction is a target, not a guarantee.
    reduction = min(max(int(decimation), 0), 99) / 100.0
    if reduction > 0 and polydata.GetNumberOfCells() > 0:
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(polydata)
        decimator.SetTargetReduction(reduction)
        decimator.PreserveTopologyOn()
        decimator.SetFeatureAngle(60)
        decimator.Update()
        polydata = decimator.GetOutput()

    # One flat colour for every cell, built in numpy and handed over once
    # rather than a Python-level SetTuple per cell. Same bytes either way; the
    # loop was only ~80ms on a mandible's 590k cells, so this is tidiness and a
    # bounded cost on the bigger structures, not a headline saving.
    cell_colors = np.tile(
        np.asarray(color_rgb, dtype=np.uint8), (polydata.GetNumberOfCells(), 1)
    )
    colors = numpy_to_vtk(cell_colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetName("Colors")
    polydata.GetCellData().SetScalars(colors)
    return polydata


def _write(polydata, output_path: str) -> None:
    import vtk

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    # `vtkPolyDataWriter` defaults to ASCII, which writes every coordinate as a
    # decimal string. Marching cubes over a CBCT at scan resolution returns
    # millions of triangles, and that default was what made AMASSS responses
    # enormous: the merged surface alone came to 848.5MB, against 6.4MB for
    # every segmentation in the same run. Binary took it to 296.7MB, and a
    # nine-structure run from 1386MB to 629MB.
    #
    # Binary is also the *more* accurate of the two, which is worth stating
    # because the reflex is to assume the opposite. It round-trips the float32
    # vertices exactly; ASCII prints them to about six significant digits, so
    # reading one back moved points by up to 5e-05mm. This writes what marching
    # cubes actually produced.
    writer.SetFileTypeToBinary()
    writer.Write()


def _meshes_in_parallel(jobs, workers: int = 1) -> list:
    """`_mesh_from_mask` over several masks at once, IN THE ORDER GIVEN.

    A mesh is a third contour and two thirds decimation, all of it C++ that
    releases the GIL -- so threads, not processes, and no 96 MB array crosses
    a process boundary to get there.

    The width is the caller's: a run holds the share admission reserved for it
    for its whole life, so spreading over that same number during the assembly
    phase spends nothing that was not already set aside. One is the honest
    default for a tool nobody granted anything.
    """
    from concurrent import futures

    jobs = list(jobs)
    width = max(1, min(int(workers or 1), len(jobs)))
    if width == 1 or len(jobs) < 2:
        return [_mesh_from_mask(*job) for job in jobs]
    with futures.ThreadPoolExecutor(max_workers=width) as pool:
        # `map` keeps the input order, which is what makes the appended file
        # reproducible whatever order the threads finish in.
        return list(pool.map(lambda job: _mesh_from_mask(*job), jobs))


def write_separate_surfaces(masks, reference, label_colors: dict, labels: dict,
                            smoothing: int, decimation: int, output_dir: str,
                            name_of, workers: int = 1) -> list:
    """Every structure's own .vtk, built side by side and written in order.

    The plural exists because the meshes are what cost: building them one at a
    time left the machine on one core for two thirds of a run's wall clock,
    with the card already idle. `masks` is an ordered mapping, and the list
    returned follows it -- so the report names the files in the run's own
    order whatever order the threads finished in.
    """
    import os

    codes = list(masks)
    meshes = _meshes_in_parallel(
        [(masks[code], reference, smoothing,
          label_colors.get(labels.get(code), (255, 255, 255)), decimation)
         for code in codes],
        workers,
    )
    written = []
    for code, polydata in zip(codes, meshes):
        path = os.path.join(output_dir, name_of(code))
        _write(polydata, path)
        logger.info("Wrote surface for %s (%d triangles)", code, polydata.GetNumberOfCells())
        written.append(path)
    return written


def write_separate_surface(mask, reference, structure_code: str,
                           label_colors: dict, labels: dict,
                           smoothing: int, output_path: str, decimation: int = 0) -> str:
    """One binary structure -> one .vtk.

    `structure_code` is passed in by the caller instead of being parsed back
    out of the file name -- this is the fix for the original KeyError.
    """
    label_index = labels.get(structure_code)
    color = label_colors.get(label_index, (255, 255, 255))
    polydata = _mesh_from_mask(mask, reference, smoothing, color, decimation)
    _write(polydata, output_path)
    logger.info(
        "Wrote surface for %s (%d triangles)", structure_code, polydata.GetNumberOfCells()
    )
    return output_path


def write_merged_surface(merged, reference, names_from_labels: dict,
                         label_colors: dict, smoothing: int,
                         output_path: str, decimation: int = 0,
                         workers: int = 1) -> str:
    """A multi-label volume -> one .vtk holding every structure's surface.

    One mesh per label, and they are built side by side: VTK's filters are
    C++ and release the GIL, measured here at x3.4 on four threads. They are
    APPENDED in label order all the same, so the file a clinician opens does
    not depend on which thread finished first.
    """
    import numpy as np
    import vtk

    wanted = []
    for label in sorted(int(value) for value in np.unique(merged)):
        if label == 0:
            continue
        structure_code = names_from_labels.get(label)
        if structure_code is None:
            # Unknown label: skip it rather than raise. The original indexed
            # NAMES_FROM_LABELS directly and died on anything unexpected.
            logger.warning("Skipping unknown label %s while building merged surface", label)
            continue
        wanted.append(label)

    meshes = _meshes_in_parallel(
        [((merged == label), reference, smoothing,
          label_colors.get(label, (255, 255, 255)), decimation) for label in wanted],
        workers,
    )
    append = vtk.vtkAppendPolyData()
    for mesh in meshes:
        append.AddInputData(mesh)
    surfaces = len(meshes)

    if surfaces == 0:
        logger.warning("No labels found in merged volume; no surface written")
        return ""

    append.Update()
    _write(append.GetOutput(), output_path)
    logger.info("Wrote merged surface with %d structure(s)", surfaces)
    return output_path

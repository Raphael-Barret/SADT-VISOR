"""Reading an intraoral mesh and turning it into the tensors the network wants.

No pytorch3d and no checkpoint here: this is the half of the engine that runs
on vtk and torch alone, and it is where three of the port's corrections live --
the label array is required rather than defaulted to zeros, the accepted
extensions are the ones discovery actually walks, and a face is kept for a
tooth only when it really touches it.
"""

import os

import numpy as np
import pytest

from conftest import write_surface
from sadt_ali_ios import surface


def test_the_readable_extensions_are_the_ones_discovery_walks(tmp_path):
    """VTK reads `.vtp`, `.obj` and `.off` too, and the original module's
    reader listed them -- but discovery never reached them. An
    accepted-then-ignored format is exactly the trap `.stl` fell into."""
    path = tmp_path / "arch.obj"
    path.write_text("v 0 0 0\n")
    with pytest.raises(ValueError, match=r"\.obj"):
        surface.read_surface(str(path))


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.vtk"):
        surface.read_surface(str(tmp_path / "nowhere.vtk"))


def test_a_file_with_no_points_is_refused_rather_than_segmented(tmp_path):
    """A `.vtk` the reader cannot make sense of comes back as an empty
    polydata, not as an error. Left unchecked it becomes an empty label array
    and a run that reports no landmarks with no reason."""
    path = tmp_path / "broken.vtk"
    path.write_text("this is not a polydata\n")
    with pytest.raises(ValueError, match="no points"):
        surface.read_surface(str(path))


@pytest.mark.parametrize("name", surface.LABEL_ARRAY_NAMES)
def test_every_label_array_name_crown_seg_writes_is_recognised(tmp_path, name):
    """The three names Crown_Seg writes. Recognising two of three would refuse
    a batch the chain had just produced."""
    mesh = write_surface(tmp_path / f"{name}.vtk", array_name=name)
    assert surface.label_array_name(surface.read_surface(mesh)) == name


def test_the_label_array_names_are_tried_in_declaration_order(tmp_path):
    """Two names on one mesh is not hypothetical -- a re-segmented mesh carries
    the older array beside the new one -- so the order has to be a decision."""
    assert surface.LABEL_ARRAY_NAMES[0] == "PredictedID"


def test_an_unlabelled_mesh_is_reported_as_carrying_no_labels(tmp_path):
    mesh = write_surface(tmp_path / "raw.vtk", labels=None)
    assert surface.label_array_name(surface.read_surface(mesh)) is None


def test_scaling_into_the_unit_sphere_is_undone_exactly_by_upscale(tmp_path):
    """The landmark is found in unit-sphere space and reported in the
    patient's own millimetres. A drift here moves every point of every run."""
    mesh = surface.read_surface(write_surface(tmp_path / "arch.vtk"))
    scaled, center, factor = surface.scale_to_unit(mesh)

    for index in range(mesh.GetNumberOfPoints()):
        back = surface.upscale(scaled.GetPoint(index), center, factor)
        assert back == pytest.approx(mesh.GetPoint(index), abs=1e-5)


def test_scaling_does_not_touch_the_mesh_it_was_given(tmp_path):
    """It deep-copies: the raw surface is still what `require_labels` and the
    locator read afterwards."""
    mesh = surface.read_surface(write_surface(tmp_path / "arch.vtk"))
    before = [mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())]
    surface.scale_to_unit(mesh)
    after = [mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())]
    assert before == after


def test_surface_properties_returns_batched_tensors(tmp_path):
    mesh = surface.read_surface(write_surface(tmp_path / "arch.vtk", labels=(8,) * 6))
    scaled, _center, _factor = surface.scale_to_unit(mesh)
    vertices, faces, colors, labels = surface.surface_properties(scaled, "cpu")

    assert vertices.shape == (1, 6, 3)
    assert faces.shape == (1, 4, 3)
    assert colors.shape == (1, 6, 3)
    assert labels.shape == (1, 6)


def test_normals_are_encoded_into_the_colour_channels_the_network_expects(tmp_path):
    """The network's first three input channels are the per-point normals
    mapped into [0, 1]. That encoding is part of what the shipped weights were
    trained against, so it is reproduced rather than chosen."""
    mesh = surface.read_surface(write_surface(tmp_path / "arch.vtk"))
    scaled, _center, _factor = surface.scale_to_unit(mesh)
    _vertices, _faces, colors, _labels = surface.surface_properties(scaled, "cpu")
    assert float(colors.min()) >= 0.0 and float(colors.max()) <= 1.0


def test_an_unlabelled_mesh_raises_instead_of_becoming_all_zeros(tmp_path):
    """Zeros mean "every vertex is tooth 0", so no tooth is ever found and the
    run ends reporting no landmarks and no reason. The engine checks the whole
    batch up front; this is the backstop under it."""
    mesh = surface.read_surface(write_surface(tmp_path / "raw.vtk", labels=None))
    scaled, _center, _factor = surface.scale_to_unit(mesh)
    with pytest.raises(ValueError) as raised:
        surface.surface_properties(scaled, "cpu")
    for name in surface.LABEL_ARRAY_NAMES:
        assert name in str(raised.value)


def test_negative_labels_are_clamped_to_the_background(tmp_path):
    """Crown_Seg writes -1 for "no tooth here". Left negative it indexes the
    label table from the end."""
    mesh = surface.read_surface(
        write_surface(tmp_path / "arch.vtk", labels=(-1, -1, 8, 8, 8, 8))
    )
    scaled, _center, _factor = surface.scale_to_unit(mesh)
    _vertices, _faces, _colors, labels = surface.surface_properties(scaled, "cpu")
    assert int(labels.min()) == 0


def test_faces_on_tooth_keeps_only_the_faces_that_touch_it(tmp_path):
    """The network predicts on a rendered view, so a mask spills onto the
    neighbouring tooth and the gum. A face is kept when at least one of its
    vertices carries this tooth's label."""
    mesh = surface.read_surface(
        write_surface(tmp_path / "arch.vtk", labels=(8, 8, 8, 9, 9, 9))
    )
    scaled, _center, _factor = surface.scale_to_unit(mesh)
    _vertices, faces, _colors, labels = surface.surface_properties(scaled, "cpu")

    every_face = list(range(faces.shape[1]))
    on_eight = surface.faces_on_tooth(faces, every_face, labels, 8)
    on_nine = surface.faces_on_tooth(faces, every_face, labels, 9)

    # Face 0 is (0, 1, 2), all tooth 8; face 3 is (3, 4, 5), all tooth 9.
    assert 0 in on_eight and 0 not in on_nine
    assert 3 in on_nine and 3 not in on_eight
    assert surface.faces_on_tooth(faces, every_face, labels, 31) == []


def test_upscale_accepts_a_plain_tuple_from_vtk(tmp_path):
    """`GetPoint` returns a tuple, not an array; the arithmetic has to promote
    it rather than concatenate it."""
    moved = surface.upscale((1.0, 2.0, 3.0), np.array([10.0, 0.0, 0.0]), 0.5)
    assert moved == pytest.approx([12.0, 4.0, 6.0])


def test_nothing_is_written_beside_the_mesh_it_read(tmp_path, monkeypatch):
    """Reading and scaling are pure. The original wrote a segmentation CSV into
    the extension's own source tree while doing it."""
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "work"
    os.makedirs(workspace)
    mesh = write_surface(workspace / "arch.vtk")
    before = sorted(os.listdir(workspace))

    scaled, _center, _factor = surface.scale_to_unit(surface.read_surface(mesh))
    surface.surface_properties(scaled, "cpu")

    assert sorted(os.listdir(workspace)) == before
    assert list(tmp_path.glob("*.csv")) == []


def test_faces_on_tooth_hands_its_faces_back_in_the_order_it_was_given(tmp_path):
    """The order is load-bearing, not tidiness.

    `_landmark_position` sums these faces' vertices one after another, and
    float32 addition is not associative -- so reordering the list moves the
    centroid in its last bits, and a centroid is snapped to the nearest mesh
    point. This pins the property that survived vectorising the filter: the
    faces come back in the order they went in, minus the ones that do not touch
    this tooth.
    """
    mesh = surface.read_surface(
        write_surface(tmp_path / "arch.vtk", labels=(8, 8, 8, 9, 9, 9))
    )
    scaled, _center, _factor = surface.scale_to_unit(mesh)
    _vertices, faces, _colors, labels = surface.surface_properties(scaled, "cpu")

    # Shuffled, and with one face repeated: a mask projects the same face once
    # per pixel that predicted it, and those duplicates are part of the sum.
    asked = [3, 0, 1, 0, 2]
    kept = surface.faces_on_tooth(faces, asked, labels, 8)

    assert kept == [face for face in asked if face in kept]
    assert kept.count(0) == 2

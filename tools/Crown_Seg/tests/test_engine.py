"""The engine, without a GPU and without the published checkpoint.

Everything here runs on a synthetic sphere and a randomly initialised network:
what is pinned is the arithmetic around the network (the camera rig, the unit
scaling, the vote, the vertex write, the post-processing, the checkpoint
loading), which is where shapeaxi's results come from and where its
non-determinism came from.

Needs monai and embreex, which are ordinary dependencies of the tool; a venv
that lacks them skips this file rather than failing it.
"""

import hashlib
import logging
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("embreex")
vtk = pytest.importorskip("vtk")

from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402

from sadt_crownseg import engine  # noqa: E402
from sadt_crownseg.errors import ToolInputError  # noqa: E402

SMALL = 64  # image side for the tests that run the network; the real one is 320


def sphere_file(path, center=(5.0, -3.0, 2.0), radius=20.0, resolution=24):
    source = vtk.vtkSphereSource()
    source.SetCenter(*center)
    source.SetRadius(radius)
    source.SetThetaResolution(resolution)
    source.SetPhiResolution(resolution)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(source.GetOutputPort())
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(triangles.GetOutputPort())
    writer.Write()
    return str(path)


@pytest.fixture(scope="module")
def sphere(tmp_path_factory):
    return sphere_file(tmp_path_factory.mktemp("sphere") / "sphere.vtk")


def random_network(seed=0):
    torch.manual_seed(seed)
    return engine.create_network().eval()


# ---------------------------------------------------------------------------
# The camera rig
# ---------------------------------------------------------------------------

def test_the_rig_is_42_cameras_on_a_sphere_of_radius_1_35():
    rig = engine.camera_rig()

    assert rig.shape == (42, 3)
    assert rig.dtype == torch.float32
    assert torch.allclose(rig.norm(dim=1), torch.full((42,), 1.35), atol=1e-6)
    assert len({tuple(row) for row in rig.tolist()}) == 42


def test_the_rig_is_bit_identical_whatever_the_global_generator_did():
    """shapeaxi nudges the on-axis cameras from the GLOBAL generator, so two
    runs of it can render the pole views differently. Here the nudge has its own
    seeded generator and the global one is never read."""
    torch.manual_seed(1)
    first = engine.camera_rig()
    torch.manual_seed(2)
    torch.rand(5)
    second = engine.camera_rig()

    assert torch.equal(first, second)


def test_the_cameras_shapeaxi_nudges_are_the_ones_this_rig_nudges():
    """shapeaxi's test is `abs(sum(v)) == radius`, which catches the six axis
    cameras and, by coincidence of the arithmetic, two others (measured: 8 of
    42). The same test, on the same cameras, is what is kept."""
    from sadt_crownseg import _shapeaxi as sx

    raw, _faces = sx.PolyDataToTensors_v_f(sx.CreateIcosahedronSubdivided(engine.RADIUS, 2))
    raw = raw.to(torch.float32)
    nudged = torch.abs(raw.sum(dim=1)) == engine.RADIUS

    plain = engine.camera_rig(seed=0)
    other = engine.camera_rig(seed=1)

    assert int(nudged.sum()) == 8
    assert torch.equal((plain != raw).any(dim=1), nudged)
    assert torch.equal(plain[~nudged], other[~nudged])
    assert not torch.equal(plain[nudged], other[nudged])
    # The nudge is a rounding error of the position: 1e-7 on a radius of 1.35.
    assert float((plain - raw).abs().max()) < 1e-6


def test_every_off_axis_camera_gets_an_orthonormal_look_at_rotation():
    rig = engine.camera_rig()
    on_axis = (rig.abs() > 1.3499).any(dim=1)
    identity = torch.eye(3)

    for position in rig[~on_axis]:
        rotation, translation = engine.camera_pose(position[None])
        assert torch.allclose(rotation[0].T @ rotation[0], identity, atol=1e-5)
        # The camera looks at the origin: the origin sits at depth |position|.
        assert torch.allclose(translation[0, 2], position.norm(), atol=1e-4)


def test_the_poses_are_pytorch3d_s_when_it_is_installed():
    """The one place pytorch3d may be named: it is the reference. A venv without
    it skips, and nothing else in this suite imports it."""
    cameras = pytest.importorskip("pytorch3d.renderer.cameras")
    rig = engine.camera_rig()

    for position in rig:
        rotation, translation = engine.camera_pose(position[None])
        expected = cameras.look_at_rotation(position[None])
        expected_t = -torch.bmm(expected.transpose(1, 2), position[None, :, None])[:, :, 0]
        assert torch.equal(rotation, expected)
        assert torch.equal(translation, expected_t)


# ---------------------------------------------------------------------------
# Input and rendering
# ---------------------------------------------------------------------------

def test_the_scan_is_scaled_into_the_unit_sphere_and_coloured_by_its_normals(sphere):
    surf, verts, faces, colors = engine.load_surface(sphere)

    assert verts.shape[1] == 3 and faces.shape[1] == 3 and colors.shape == verts.shape
    # Centred on the middle of the bounding box and scaled by the inverse of its
    # half-diagonal, so every point is inside the unit sphere (a sphere of
    # radius 20 ends at 1/sqrt(3), the box corner being the one that touches).
    assert float(verts.norm(dim=1).max()) <= 1.0 + 1e-5
    assert float(verts.norm(dim=1).max()) == pytest.approx(3 ** -0.5, abs=2e-2)
    assert float((verts.max(dim=0).values + verts.min(dim=0).values).abs().max()) < 1e-5
    assert float(colors.min()) >= 0.0 and float(colors.max()) <= 1.0
    # The surface that gets labelled and written keeps the scan's millimetres.
    original = vtk_to_numpy(surf.GetPoints().GetData())
    assert float(np.abs(original).max()) > 5.0


def test_a_view_is_normals_and_depth_on_a_white_background(sphere):
    _surf, verts, faces, colors = engine.load_surface(sphere)
    views, pix_to_face = engine.render_views(verts, faces, colors, engine.camera_rig()[:3], image_size=SMALL)

    assert views.shape == (3, 4, SMALL, SMALL)
    assert pix_to_face.shape == (3, SMALL, SMALL)
    hit = pix_to_face >= 0
    assert 0.05 < float(hit.float().mean()) < 0.9
    background = ~hit[:, None].expand(-1, 4, -1, -1)
    # White around the arch, and depth -1 where nothing was hit: the two
    # conventions the network was trained on.
    assert bool((views[:, :3][background[:, :3]] == 1.0).all())
    assert bool((views[:, 3][~hit] == -1.0).all())
    assert bool((views[:, 3][hit] > 0).all())
    assert int(pix_to_face.max()) < faces.shape[0]


def test_the_views_are_bit_identical_from_one_call_to_the_next(sphere):
    _surf, verts, faces, colors = engine.load_surface(sphere)
    rig = engine.camera_rig()

    first = engine.render_views(verts, faces, colors, rig, image_size=SMALL)
    second = engine.render_views(verts, faces, colors, rig, image_size=SMALL)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


# ---------------------------------------------------------------------------
# The vote
# ---------------------------------------------------------------------------

def test_a_face_hit_by_several_pixels_keeps_the_last_ones_value_as_shapeaxi_does():
    totals = np.zeros((2, 3), dtype=np.float32)
    pix_to_face = np.array([1, 1, 0])
    values = np.array([[0.1, 0.2, 0.5], [0.9, 0.8, 0.5]], dtype=np.float32)

    engine.accumulate_view(totals, pix_to_face, values)

    assert totals[:, 1].tolist() == pytest.approx([0.2, 0.8])
    assert totals[:, 0].tolist() == pytest.approx([0.5, 0.5])


def test_a_pixel_that_missed_lands_on_the_last_face_as_it_does_in_shapeaxi():
    """-1 indexes the last face there too. The softmax of zeroed logits is
    uniform, so it adds the same amount to every class and cannot change an
    argmax; it is kept because dropping it would be a change of behaviour."""
    totals = np.zeros((2, 3), dtype=np.float32)

    engine.accumulate_view(totals, np.array([-1]), np.array([[0.25], [0.25]], dtype=np.float32))

    assert totals[:, 2].tolist() == [0.25, 0.25]


def test_accumulating_pixels_sums_them_and_ignores_the_background():
    totals = np.zeros((2, 3), dtype=np.float32)
    pix_to_face = np.array([1, 1, 0, -1])
    values = np.array([[0.1, 0.2, 0.5, 0.25], [0.9, 0.8, 0.5, 0.25]], dtype=np.float32)

    engine.accumulate_view(totals, pix_to_face, values, accumulate_pixels=True)

    assert totals[:, 1].tolist() == pytest.approx([0.3, 1.7])
    assert totals[:, 2].tolist() == [0.0, 0.0]


def test_views_are_added_to_each_other():
    totals = np.zeros((2, 2), dtype=np.float32)
    for value in (0.25, 0.5):
        engine.accumulate_view(totals, np.array([0]), np.array([[value], [0.0]], dtype=np.float32))

    assert totals[0, 0] == pytest.approx(0.75)


def test_the_vote_is_what_torch_computes_on_one_cpu_thread():
    """shapeaxi's own lines, run by torch on the CPU, against the numpy version:
    the claim is that this reproduces shapeaxi's CPU result, so it is compared
    with it rather than assumed. With repeated indices, as a real view has."""
    generator = np.random.default_rng(0)
    faces, pixels = 200, 5000
    pix_to_face = generator.integers(-1, faces, size=pixels)
    values = generator.random((engine.OUT_CHANNELS, pixels), dtype=np.float32)

    ours = np.zeros((engine.OUT_CHANNELS, faces), dtype=np.float32)
    engine.accumulate_view(ours, pix_to_face, values)

    # ONE thread, because that is the only case where torch's own answer is
    # defined: with the default thread pool the same line gives a different
    # array on each call (measured here with 28 threads: three calls, three
    # results). That race on the CPU is part of why shapeaxi disagrees with
    # itself, and it is what the numpy write removes.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        theirs = torch.zeros(engine.OUT_CHANNELS, faces)
        theirs[:, torch.from_numpy(pix_to_face)] += torch.from_numpy(values)
    finally:
        torch.set_num_threads(threads)

    assert np.array_equal(ours, theirs.numpy())


def test_a_vertex_takes_the_label_of_the_last_face_that_starts_at_it():
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]])

    labels = engine.vertex_labels(faces, np.array([5, 7, 3]), vertex_count=5)

    assert labels.tolist() == [7, 3, 0, 0, 0]
    assert labels.dtype == np.int64


def test_the_vertex_write_is_what_torch_computes_on_one_cpu_thread():
    generator = np.random.default_rng(1)
    vertices = 300
    faces = generator.integers(0, vertices, size=(1200, 3))
    face_labels = generator.integers(0, engine.OUT_CHANNELS, size=1200)

    ours = engine.vertex_labels(faces, face_labels, vertices)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # see the vote test above: torch's own answer is defined for one thread only
    try:
        theirs = torch.zeros(vertices, dtype=torch.int64)
        theirs[torch.from_numpy(faces)[:, 0]] = torch.from_numpy(face_labels)
    finally:
        torch.set_num_threads(threads)

    assert np.array_equal(ours, theirs.numpy())


# ---------------------------------------------------------------------------
# Post-processing and numbering
# ---------------------------------------------------------------------------

def test_a_tiny_island_is_absorbed_by_its_neighbours(sphere):
    from vtk.util.numpy_support import numpy_to_vtk

    surf, _verts, _faces, _colors = engine.load_surface(sphere)
    count = surf.GetNumberOfPoints()
    labels = np.full(count, engine.GINGIVA_LABEL, dtype=np.int64)
    labels[100:104] = 8   # four points: far below the 200 an island must have
    array = numpy_to_vtk(labels)
    array.SetName("Universal_ID")
    surf.GetPointData().AddArray(array)

    engine.post_process(surf, array)

    assert 8 not in set(vtk_to_numpy(surf.GetPointData().GetScalars("Universal_ID")).tolist())


def test_fdi_renumbering_is_shapeaxis_table():
    from vtk.util.numpy_support import numpy_to_vtk

    surf = vtk.vtkPolyData()
    array = numpy_to_vtk(np.array([0, 1, 8, 9, 16, 17, 24, 25, 32, 33], dtype=np.int64))
    array.SetName("Universal_ID")
    surf.GetPointData().AddArray(array)

    engine.to_fdi(surf, "Universal_ID")

    assert vtk_to_numpy(surf.GetPointData().GetScalars("Universal_ID")).tolist() == [
        0, 18, 11, 21, 28, 38, 31, 41, 48, 0,
    ]


# ---------------------------------------------------------------------------
# Loading the weights
# ---------------------------------------------------------------------------

def save_state(path, network=None, transform=None):
    state = (network or random_network()).state_dict()
    if transform:
        state = transform(state)
    torch.save(state, path)
    return path


def test_a_plain_state_dict_loads_strictly_and_gives_the_same_weights(tmp_path):
    network = random_network(3)
    path = save_state(tmp_path / "a_val-loss0.1.pth", network)

    loaded = engine.load_network(path)

    for name, tensor in network.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)


def test_a_checkpoint_wrapped_in_state_dict_loads_too(tmp_path):
    path = tmp_path / "a_val-loss0.1.pth"
    torch.save({"state_dict": random_network().state_dict()}, path)

    engine.load_network(path)


def test_a_file_named_like_the_published_one_must_be_the_published_one(tmp_path):
    path = save_state(tmp_path / engine.PUBLISHED_CHECKPOINT)

    with pytest.raises(ToolInputError) as caught:
        engine.load_network(path)

    assert "not the published" in str(caught.value)
    assert engine.PUBLISHED_SHA256 in str(caught.value)


def test_another_vintage_under_another_name_is_loaded_and_its_digest_logged(tmp_path, caplog):
    path = save_state(tmp_path / "next_val-loss0.150.pth")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with caplog.at_level(logging.INFO, logger="sadt_crownseg.engine"):
        engine.load_network(path)

    assert digest in caplog.text


def test_a_checkpoint_that_would_run_code_is_refused_not_executed(tmp_path):
    marker = tmp_path / "executed"

    class Trap:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    path = tmp_path / "trap_val-loss0.1.pth"
    torch.save({"model.weight": Trap()}, path)

    with pytest.raises(ToolInputError):
        engine.load_network(path)

    assert not marker.exists()


def test_a_state_dict_with_the_prefix_stripped_is_refused_not_loaded_as_nothing(tmp_path):
    """`strict=False` on these keys would load NOTHING and leave a random
    network that answers everything at about 0.15 confidence."""
    path = save_state(
        tmp_path / "stripped_val-loss0.1.pth",
        transform=lambda state: {key.removeprefix("model."): value for key, value in state.items()},
    )

    with pytest.raises(ToolInputError):
        engine.load_network(path)


def test_a_missing_or_truncated_checkpoint_is_an_input_error(tmp_path):
    with pytest.raises(ToolInputError):
        engine.load_network(tmp_path / "absent_val-loss0.1.pth")

    broken = tmp_path / "broken_val-loss0.1.pth"
    broken.write_bytes(b"not a checkpoint")
    with pytest.raises(ToolInputError):
        engine.load_network(broken)


@pytest.mark.models
def test_the_published_checkpoint_is_the_documented_one_and_loads_weights_only():
    path = os.environ.get("SADT_CROWNSEG_MODEL")
    if not path:
        pytest.skip("set SADT_CROWNSEG_MODEL (see tests/data/README.md)")

    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == engine.PUBLISHED_SHA256
    engine.load_network(path)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_two_runs_on_the_same_mesh_give_bit_identical_labels(sphere):
    network = random_network(0)
    rig = engine.camera_rig()
    _surf, verts, faces, colors = engine.load_surface(sphere)

    first = engine.predict(network, rig, verts, faces, colors, image_size=SMALL)
    # Whatever the global generator is doing in between must not matter.
    torch.manual_seed(12345)
    torch.rand(1000)
    second = engine.predict(network, engine.camera_rig(), verts, faces, colors, image_size=SMALL)

    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) > 1, "a constant answer would prove nothing"


def test_the_written_file_is_byte_identical_from_one_run_to_the_next(sphere, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "IMAGE_SIZE", SMALL)
    network = random_network(0)
    rig = engine.camera_rig()

    outputs = [
        engine.segment_mesh(network, rig, sphere, str(tmp_path / name / "sphere_Seg.vtk"), "Universal_ID")
        for name in ("a", "b")
    ]

    first, second = (open(path, "rb").read() for path in outputs)
    assert first == second
    # Slicer reads a .vtk without this line as LPS and warns on every load.
    assert first.splitlines()[1] == b"SPACE=LPS"
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(outputs[0])
    reader.Update()
    assert reader.GetOutput().GetPointData().GetArray("Universal_ID") is not None


def test_the_input_scan_is_never_modified(sphere, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "IMAGE_SIZE", SMALL)
    before = open(sphere, "rb").read()

    engine.segment_mesh(random_network(), engine.camera_rig(), sphere, str(tmp_path / "o" / "s.vtk"), "Universal_ID")

    assert open(sphere, "rb").read() == before


def test_fdi_numbering_writes_fdi_values_under_the_requested_name(sphere, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "IMAGE_SIZE", SMALL)
    network, rig = random_network(), engine.camera_rig()

    universal = engine.segment_mesh(network, rig, sphere, str(tmp_path / "u.vtk"), "Universal_ID")
    fdi = engine.segment_mesh(network, rig, sphere, str(tmp_path / "f.vtk"), "FDI_ID", fdi=True)

    def labels(path, name):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars(name))

    assert np.array_equal(labels(fdi, "FDI_ID"), engine.FDI_LUT[labels(universal, "Universal_ID")])

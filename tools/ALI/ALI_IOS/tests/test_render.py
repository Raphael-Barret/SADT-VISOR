"""The camera geometry, which is all of `render.py` that runs without pytorch3d.

Everything here is torch and numpy: where a tooth is, which way the arch runs
at it, where the mucogingival cameras aim and what to do about a tooth the
segmentation missed. The rasterizer itself is the only part that needs the
compiled extension, and it is exercised through the stub in `conftest.py`.

The camera DIRECTIONS are pinned rather than recomputed: the shipped weights
were trained on the views these exact vectors produce, inconsistencies
included -- several are divided by the norm of a different vector than the one
being normalised, so they are not unit length. That is not a typo to fix.
"""

import numpy as np
import pytest
import torch

from sadt_ali_ios import catalog, render


def labelled(numbers, positions=None):
    """`(labels, vertices)` as the batched tensors the engine passes around."""
    labels = torch.tensor([list(numbers)], dtype=torch.int64)
    if positions is None:
        positions = [[float(index), 0.0, 0.0] for index in range(len(numbers))]
    return labels, torch.tensor([positions], dtype=torch.float32)


# ---------------------------------------------------------------------------
# The trained rasterisation settings
# ---------------------------------------------------------------------------

def test_the_rasterisation_settings_are_the_trained_ones():
    """Fixed in the original CLI's XML and never exposed: they describe the
    trained networks, not a user preference. Changing one silently changes what
    every network sees."""
    assert (render.IMAGE_SIZE, render.BLUR_RADIUS, render.FACES_PER_PIXEL) == (224, 0, 1)


def test_the_occlusal_cameras_keep_their_upstream_inconsistency():
    """Five of the occlusal directions are divided by the norm of a DIFFERENT
    vector than the one being normalised, so they are not unit length.
    Reproduced byte for byte because the weights were trained on the views
    these exact vectors produce -- "fixing" them changes every result."""
    upper = render.CAMERA_POSITIONS["O"]["Upper"]
    assert list(upper[0]) == [0, 0, -1]
    # [0.5, 0, -1] over the norm of [0.5, 0.5, -1], which is not its own.
    assert np.linalg.norm(upper[1]) != pytest.approx(1.0)
    assert upper[1] == pytest.approx(np.array([0.5, 0.0, -1.0]) / np.linalg.norm([0.5, 0.5, -1.0]))


def test_the_cervical_cameras_orbit_the_arch_from_the_side():
    """Twelve directions, all unit length, all with a non-zero horizontal
    component: the cervical points sit on the side of the crown."""
    for jaw in ("Upper", "Lower"):
        directions = render.CAMERA_POSITIONS["C"][jaw]
        assert len(directions) == 12
        for direction in directions:
            assert np.linalg.norm(direction) == pytest.approx(1.0)
            assert abs(direction[0]) > 0


def test_the_two_jaws_look_at_the_arch_from_opposite_sides():
    """Occlusal cameras look down on a maxilla and up at a mandible. One sign
    wrong and every view is of the far side of the arch."""
    assert render.CAMERA_POSITIONS["O"]["Upper"][0][2] < 0
    assert render.CAMERA_POSITIONS["O"]["Lower"][0][2] > 0


def test_every_network_the_catalog_declares_has_a_camera_radius():
    """A network with no radius is a `KeyError` at the first tooth of a run."""
    assert set(catalog.CAMERA_RADIUS) == set(catalog.NETWORK_CODES)
    assert catalog.CAMERA_RADIUS["C"] > catalog.CAMERA_RADIUS["O"]


# ---------------------------------------------------------------------------
# tooth_center
# ---------------------------------------------------------------------------

def test_a_tooth_that_is_not_in_the_mesh_has_no_centre():
    """The original returned a zero vector, which is a real position at the
    centre of the arch -- so the cameras dutifully rendered the middle of the
    palate and the network was asked to find a cusp in it."""
    labels, vertices = labelled([8, 8, 9, 9])
    assert render.tooth_center(labels, vertices, 30, "cpu") is None


def test_a_tooth_centre_is_the_centroid_of_its_own_vertices():
    labels, vertices = labelled(
        [8, 8, 9, 9],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 4.0, 0.0]],
    )
    center = render.tooth_center(labels, vertices, 8, "cpu")
    assert center.shape == (1, 3)
    assert center[0].tolist() == pytest.approx([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# arch_tangent
# ---------------------------------------------------------------------------

def test_the_arch_tangent_is_taken_between_the_two_neighbouring_teeth():
    """The lower teeth carry consecutive Universal ids along the arch, so the
    neighbours of a tooth are its id minus and plus one."""
    labels, vertices = labelled(
        [24, 25, 26],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 5.0], [2.0, 0.0, 0.0]],
    )
    tangent = render.arch_tangent(labels, vertices, 25, "cpu")
    assert tangent.tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_the_arch_tangent_is_horizontal_and_unit_length():
    """It is read in the horizontal plane: the vertical component is dropped
    before normalising, so a step in height does not tilt the cameras."""
    labels, vertices = labelled(
        [24, 25, 26],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 9.0], [2.0, 2.0, -9.0]],
    )
    tangent = render.arch_tangent(labels, vertices, 25, "cpu")
    assert float(tangent[2]) == 0.0
    assert float(torch.norm(tangent)) == pytest.approx(1.0, abs=1e-5)


def test_one_sided_at_the_end_of_the_arch():
    """The last molar has no neighbour beyond it; a one-sided difference is
    used rather than returning nothing."""
    labels, vertices = labelled([30, 31], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tangent = render.arch_tangent(labels, vertices, 31, "cpu")
    assert tangent.tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_no_neighbour_at_all_means_no_tangent():
    """The caller then falls back to the radial direction, which is what the
    sphere scheme used everywhere."""
    labels, vertices = labelled([25], [[0.0, 0.0, 0.0]])
    assert render.arch_tangent(labels, vertices, 25, "cpu") is None


def test_two_coincident_neighbours_give_no_tangent_rather_than_a_nan():
    """Normalising a zero-length vector is how a NaN gets into a rotation
    matrix and out into a landmark position."""
    labels, vertices = labelled([24, 25, 26], [[1.0, 1.0, 0.0]] * 3)
    assert render.arch_tangent(labels, vertices, 25, "cpu") is None


# ---------------------------------------------------------------------------
# mg_frame
# ---------------------------------------------------------------------------

def test_the_buccal_normal_is_horizontal_perpendicular_and_points_outward():
    """At the cheek, not the tongue. Pointing it inward renders the lingual
    surface and the mucogingival point is not in the image at all."""
    labels, vertices = labelled(
        [24, 25, 26],
        [[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
    )
    center = render.tooth_center(labels, vertices, 25, "cpu")
    tangent = render.arch_tangent(labels, vertices, 25, "cpu")
    normal, _aim = render.mg_frame(vertices, center, tangent, 25, "cpu")

    assert float(normal[2]) == pytest.approx(0.0)
    assert float(torch.norm(normal)) == pytest.approx(1.0, abs=1e-4)
    assert float(torch.dot(normal, tangent)) == pytest.approx(0.0, abs=1e-5)
    # Away from the arch's own centre, which sits below tooth 25 here.
    outward = center[0] - vertices[0].mean(dim=0)
    assert float(torch.dot(normal[:2], outward[:2])) > 0


def test_the_cameras_aim_at_the_landmark_not_at_a_flat_drop():
    """The radial direction the sphere scheme used is only buccal near the
    midline -- 35 degrees off on teeth 19/30 and 53 on tooth 31, so on the
    molars the cameras looked ALONG the arch and the landmark was outside the
    render. The offsets are a measured anatomical prior."""
    labels, vertices = labelled(
        [30, 31], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    center = render.tooth_center(labels, vertices, 31, "cpu")
    tangent = render.arch_tangent(labels, vertices, 31, "cpu")
    _normal, aim = render.mg_frame(vertices, center, tangent, 31, "cpu")

    buccal, _along, vertical = catalog.MG_AIM_OFFSET[31]
    assert float(aim[2]) == pytest.approx(float(center[0][2]) + vertical, abs=1e-5)
    assert buccal > 0.1, "tooth 31 is the one the flat drop missed by most"


def test_a_tooth_with_no_tangent_falls_back_to_the_flat_drop():
    """What the code did before the offsets were measured, kept for a tooth
    whose neighbours are absent: the offsets are expressed in the tooth's own
    frame and there is no frame without a tangent."""
    labels, vertices = labelled([25], [[0.0, 0.0, 4.0]])
    center = render.tooth_center(labels, vertices, 25, "cpu")
    _normal, aim = render.mg_frame(vertices, center, None, 25, "cpu")
    assert float(aim[2]) == pytest.approx(4.0 - 0.2)


def test_the_three_mucogingival_cameras_stay_horizontal_and_lead_with_the_normal():
    """The network sees three images in a fixed order; a different geometry is
    a different input. Front first, then the two rotated about the vertical."""
    normal = torch.tensor([1.0, 0.0, 0.0])
    directions = render.mg_camera_directions(normal, "cpu")

    assert directions.shape == (3, 3)
    assert directions[0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    for direction in directions:
        assert float(direction[2]) == 0.0
    # Rotated by +/- the spread, so the two flanks are mirror images.
    assert float(directions[1][1]) == pytest.approx(-float(directions[2][1]))
    assert float(directions[1][1]) == pytest.approx(np.sin(render.MG_CAMERA_SPREAD))


# ---------------------------------------------------------------------------
# estimate_missing_teeth
# ---------------------------------------------------------------------------

def _arch(numbers):
    """Centroids on a parabola through the ids, one point per tooth."""
    positions = [[float(n), float(n) ** 2 / 40.0, 1.0] for n in numbers]
    return labelled(numbers, positions)


def test_nothing_is_estimated_when_every_wanted_tooth_is_segmented():
    labels, vertices = _arch(list(catalog.MG_TEETH))
    assert render.estimate_missing_teeth(labels, vertices, catalog.MG_TEETH, "cpu") == {}


def test_a_gap_in_the_segmentation_is_filled_from_the_arch():
    """A gap in the segmentation must not become a gap in the mucogingival
    line, which is what AREG then has to fit a spline through."""
    present = [n for n in catalog.MG_TEETH if n != 25]
    labels, vertices = _arch(present)

    estimates = render.estimate_missing_teeth(labels, vertices, catalog.MG_TEETH, "cpu")
    assert set(estimates) == {25}

    center, tangent = estimates[25]
    assert center.tolist() == pytest.approx([25.0, 25.0 ** 2 / 40.0, 1.0], abs=1e-2)
    # The tangent runs along the arch, so it leads in the id direction.
    assert float(tangent[0]) > 0


def test_too_few_segmented_teeth_is_no_estimate_rather_than_an_extrapolation():
    """Below four teeth, or a span below four ids, the quadratic fit is not
    trustworthy -- and a confidently wrong camera aim is worse than a skipped
    tooth."""
    labels, vertices = _arch([19, 20, 21])
    assert render.estimate_missing_teeth(labels, vertices, catalog.MG_TEETH, "cpu") == {}


def test_four_teeth_crowded_into_a_short_span_is_also_refused():
    labels, vertices = _arch([19, 20, 21, 22])
    assert render.estimate_missing_teeth(labels, vertices, catalog.MG_TEETH, "cpu") == {}
    # One more id of span and the same four-tooth count is accepted.
    labels, vertices = _arch([19, 20, 21, 23])
    assert render.estimate_missing_teeth(labels, vertices, catalog.MG_TEETH, "cpu")


# ---------------------------------------------------------------------------
# One rasterization per view
# ---------------------------------------------------------------------------

class _CountingRenderer:
    """A renderer shaped like `MeshRendererWithFragments`: it hands back the
    fragments it rasterized, and it counts how often it was asked.

    `rasterizer` is here so a test can prove nothing reaches it. Calling it
    raises: a second rasterization is the defect these two tests exist to
    catch, and it is invisible in the output -- the images come out identical,
    the run just takes twice as long in its most expensive phase.
    """

    def __init__(self, pixels: int = 2):
        self.calls = []
        self.pixels = pixels

    class _Rasterizer:
        def __call__(self, *args, **kwargs):
            raise AssertionError("a view was rasterized a second time")

    rasterizer = _Rasterizer()

    def __call__(self, meshes_world, R=None, T=None):
        self.calls.append((R, T))
        size = self.pixels
        # (batch, H, W, RGBA), which the callers permute and trim to RGB.
        image = torch.zeros(1, size, size, 4)
        fragments = type(
            "Fragments",
            (),
            {
                "zbuf": torch.zeros(1, size, size, 1),
                "pix_to_face": torch.zeros(1, size, size, 1, dtype=torch.int64),
            },
        )()
        return image, fragments


class _PlainMesh:
    def clone(self):
        return self


def test_a_crown_view_is_rasterized_once_and_the_fragments_come_back_with_it():
    """The depth channel and `pix_to_face` both come from the fragments, and
    they used to be recovered by rasterizing the mesh AGAIN -- 238 views of a
    real mesh, 476 rasterizations, half of them recomputing what the shading
    pass had just thrown away.
    """
    renderer = _CountingRenderer()
    directions = render.CAMERA_POSITIONS["C"]["Upper"]

    images, pix_to_face = render.render_views(
        renderer=renderer, mesh=_PlainMesh(), center=torch.zeros(1, 3),
        radius=1.0, camera_positions=directions, device="cpu",
    )

    assert len(renderer.calls) == len(directions)
    # Every call carried its own pose. The second rasterization used to pass
    # none at all, reading the pose out of the cameras the first had left it in.
    assert all(R is not None and T is not None for R, T in renderer.calls)
    assert images.shape[1] == len(directions)
    assert pix_to_face.shape[0] == len(directions)


def test_a_mucogingival_view_is_rasterized_once_too():
    """The same fix on the other path, which AREG_IOSCBCT is the only caller
    of -- so it is the one least likely to be noticed by hand."""
    renderer = _CountingRenderer()
    normal = torch.tensor([1.0, 0.0, 0.0])
    directions = render.mg_camera_directions(normal, "cpu")

    images, pix_to_face = render.render_mg_views(
        renderer=renderer, mesh=_PlainMesh(), aim=torch.zeros(3),
        directions=directions, radius=1.0, device="cpu",
    )

    assert len(renderer.calls) == 3
    assert images.shape[1] == 3
    assert pix_to_face.shape[0] == 3

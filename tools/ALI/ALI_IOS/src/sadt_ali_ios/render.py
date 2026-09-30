"""Offscreen rendering of a tooth from several viewpoints.

The IOS engine does not look at the mesh directly: it renders the tooth from a
fixed set of camera directions, runs a 2D UNet on each image, and projects the
predicted mask back onto the faces that produced those pixels. This module
builds the renderer and holds the camera directions.

Ported from ALI_IOS_utils/{render,mask_renderer,agent}.py. The camera
directions are reproduced byte for byte INCLUDING their inconsistencies:
several are divided by the norm of a different vector than the one being
normalized, so they are not unit length. That is not a typo to fix -- the
shipped weights were trained on the views these exact vectors produce.

pytorch3d is imported inside the functions that need it. It is an optional
extra of this package, compiled from source, so on a venv without it ALI still
imports and publishes its schema, and only an IOS run fails.
"""

import math

import numpy as np

from .errors import ToolUnavailableError

_INSTALL_HINT = (
    "ALI's IOS engine needs pytorch3d, which publishes no usable wheel and is "
    "compiled from source against this venv's torch. See the 'pytorch3d' section "
    "of tools/ALI/README.md. The CBCT engine works without it."
)


def _norm(vector) -> float:
    return float(np.linalg.norm(vector))


# Camera directions per network and jaw, verbatim from the original
# `dic_cam`. The occlusal ones look straight down (or up) the arch and fan out
# slightly; the cervical ones orbit it from the side, where those points sit.
CAMERA_POSITIONS = {
    "O": {
        "Upper": (
            [0, 0, -1],
            np.array([0.5, 0.0, -1]) / _norm([0.5, 0.5, -1]),
            np.array([-0.5, 0.0, -1]) / _norm([-0.5, -0.5, -1]),
            np.array([0, 0.5, -1]) / _norm([1, 0, -1]),
            np.array([0, -0.5, -1]) / _norm([0, 1, -1]),
        ),
        "Lower": (
            [0, 0, 1],
            np.array([0.5, 0.0, 1.0]) / _norm([0.5, 0.5, 1.0]),
            np.array([-0.5, 0.0, 1.0]) / _norm([-0.5, -0.5, 1.0]),
            np.array([0, 0.5, 1]) / _norm([1, 0, 1]),
            np.array([0, -0.5, 1]) / _norm([0, 1, 1]),
        ),
    },
    "C": {
        "Upper": tuple(
            np.array(vector) / _norm(vector)
            for vector in [
                [1, 0, 0], [-1, 0, 0], [1, -1, 0], [-1, -1, 0], [1, 1, 0], [-1, 1, 0],
                [1, 0, -0.5], [-1, 0, -0.5], [1, -1, -0.5], [-1, -1, -0.5],
                [1, 1, -0.5], [-1, 1, -0.5],
            ]
        ),
        "Lower": tuple(
            np.array(vector) / _norm(vector)
            for vector in [
                [1, 0, 0], [-1, 0, 0], [1, -1, 0], [-1, -1, 0], [1, 1, 0], [-1, 1, 0],
                [1, 0, 0.5], [-1, 0, 0.5], [1, -1, 0.5], [-1, -1, 0.5],
                [1, 1, 0.5], [-1, 1, 0.5],
            ]
        ),
    },
}

# Rasterization settings the models were trained with. Fixed in the original
# CLI's XML and never exposed to the user; they describe the trained networks,
# not a user preference.
IMAGE_SIZE = 224
BLUR_RADIUS = 0
FACES_PER_PIXEL = 1


def import_pytorch3d():
    try:
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            HardPhongShader,
            MeshRasterizer,
            MeshRendererWithFragments,
            PointLights,
            RasterizationSettings,
            blending,
            look_at_rotation,
        )
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: {exc.name or 'pytorch3d'})") from exc

    return {
        "FoVPerspectiveCameras": FoVPerspectiveCameras,
        "HardPhongShader": HardPhongShader,
        "MeshRasterizer": MeshRasterizer,
        "MeshRendererWithFragments": MeshRendererWithFragments,
        "Meshes": Meshes,
        "PointLights": PointLights,
        "RasterizationSettings": RasterizationSettings,
        "TexturesVertex": TexturesVertex,
        "blending": blending,
        "look_at_rotation": look_at_rotation,
    }


def build_renderer(device, image_size: int = IMAGE_SIZE, blur_radius: float = BLUR_RADIUS,
                   faces_per_pixel: int = FACES_PER_PIXEL):
    """A Phong renderer over a 90-degree perspective camera.

    The original built a second, mask-only renderer alongside this one. It is
    gone: nothing on the inference path ever called it (only the training-time
    `GetView(rend=True)` did), and it was the sole reason `mask_renderer.py`
    existed.

    **`MeshRendererWithFragments`, not `MeshRenderer`.** The two have the same
    `forward` -- rasterize, then shade -- and differ only in that this one also
    returns the fragments it rasterized. Both callers below need them, for the
    depth channel and for `pix_to_face`, and with the plain renderer the only
    way to get them was to rasterize the mesh a SECOND time. Measured on the
    reference mesh: 238 views rendered, 476 rasterizations, 3.57 s of a 21.6 s
    run spent recomputing fragments the shading pass had just thrown away.
    """
    p3d = import_pytorch3d()

    cameras = p3d["FoVPerspectiveCameras"](znear=0.01, zfar=10, fov=90, device=device)
    raster_settings = p3d["RasterizationSettings"](
        image_size=image_size, blur_radius=blur_radius, faces_per_pixel=faces_per_pixel
    )
    rasterizer = p3d["MeshRasterizer"](cameras=cameras, raster_settings=raster_settings)
    shader = p3d["HardPhongShader"](
        device=device,
        cameras=cameras,
        lights=p3d["PointLights"](device=device),
        blend_params=p3d["blending"].BlendParams(background_color=(0, 0, 0)),
    )
    return p3d["MeshRendererWithFragments"](rasterizer=rasterizer, shader=shader)


def tooth_center(labels, vertices, tooth_number: int, device):
    """The centroid of a tooth's vertices, which the cameras orbit.

    Returns None when the tooth is not in this mesh -- the original returned
    a zero vector, which is a real position at the centre of the arch, so the
    cameras dutifully rendered the middle of the palate and the network was
    asked to find a cusp in it.
    """
    label_table = labels[0]
    matches = (label_table == int(tooth_number)).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return None
    return vertices[0][matches].mean(dim=0).unsqueeze(0).to(device)


#
# The radial direction (tooth centre - mesh centre) that the sphere scheme
# effectively uses is only buccal near the midline. Measured upstream against
# the true normal: off by 2-5 degrees on the incisors, 35 degrees on teeth
# 19/30 and 53 degrees on tooth 31 -- so on the molars the cameras looked ALONG
# the arch and the landmark was not in the image at all.

# Half-angle between the front camera and its two neighbours, in radians.
MG_CAMERA_SPREAD = 0.35
# How far the cameras drop below the aim point, as a fraction of the radius.
MG_CAMERA_PLUNGE = 0.15


def arch_tangent(labels, vertices, tooth_number: int, device):
    """The arch's direction at one tooth, horizontal and unit length.

    The lower teeth carry consecutive Universal ids along the arch (18 -> 31),
    so the neighbours of `tooth_number` are its id minus and plus one, and their
    centroids give the local tangent. At the ends of the arch one neighbour is
    missing and a one-sided difference is used. Returns None when neither
    neighbour is present -- the caller then falls back to the radial direction.
    """
    from .torch_helpers import import_torch

    torch = import_torch()

    def centroid(number):
        matches = (labels[0] == int(number)).nonzero(as_tuple=True)[0]
        return vertices[0][matches].mean(dim=0) if len(matches) else None

    here, before, after = (
        centroid(tooth_number), centroid(tooth_number - 1), centroid(tooth_number + 1)
    )
    if before is not None and after is not None:
        tangent = after - before
    elif after is not None and here is not None:
        tangent = after - here
    elif before is not None and here is not None:
        tangent = here - before
    else:
        return None

    tangent = tangent.clone()
    tangent[2] = 0.0  # the arch is read in the horizontal plane
    norm = torch.norm(tangent)
    return (tangent / norm).to(device) if norm > 1e-6 else None


def mg_frame(vertices, center, tangent, tooth_number: int, device):
    """`(buccal normal, aim point)` for one tooth, in unit-sphere space.

    The buccal normal is horizontal, perpendicular to the arch tangent, and
    pointing away from the arch's centre -- at the cheek, not the tongue. The
    aim point is where the landmark is expected (`MG_AIM_OFFSET`), so the
    cameras frame the gingival margin rather than the crown.
    """
    from .torch_helpers import import_torch

    from . import catalog

    torch = import_torch()

    outward = (center[0] - vertices[0].mean(dim=0)).clone()
    outward[2] = 0.0

    if tangent is None:
        normal = outward
    else:
        normal = torch.stack([-tangent[1], tangent[0], torch.zeros_like(tangent[0])])
        if torch.norm(normal) < 1e-6:
            normal = outward
        if torch.dot(normal, outward) < 0:
            normal = -normal
    normal = normal / (torch.norm(normal) + 1e-6)

    aim = center[0].clone()
    offset = catalog.MG_AIM_OFFSET.get(int(tooth_number))
    if offset is not None and tangent is not None:
        buccal, along, vertical = offset
        aim = aim + normal * buccal + tangent * along
        aim[2] = aim[2] + vertical
    else:
        # What the code did before the offsets were measured, kept as the
        # fallback for a tooth with no known neighbour.
        aim[2] = aim[2] - 0.2

    return normal.to(device), aim.to(device)


def mg_camera_directions(normal, device):
    """The three buccal directions: straight on, and rotated by +/- the spread.

    Rotated about the vertical axis, so all three stay horizontal. Must match
    the training code: the network sees three images in a fixed order and a
    different geometry is a different input.
    """
    from .torch_helpers import import_torch

    torch = import_torch()

    cos_a, sin_a = math.cos(MG_CAMERA_SPREAD), math.sin(MG_CAMERA_SPREAD)
    front = normal
    left = front.clone()
    right = front.clone()
    left[0] = front[0] * cos_a - front[1] * sin_a
    left[1] = front[0] * sin_a + front[1] * cos_a
    right[0] = front[0] * cos_a + front[1] * sin_a
    right[1] = -front[0] * sin_a + front[1] * cos_a
    return torch.stack([front, left, right]).to(device)


def render_mg_views(renderer, mesh, aim, directions, radius: float, device):
    """Render the three buccal views of one tooth's gingival margin.

    Returns `(images, pix_to_face)` shaped like `render_views`' output, so the
    engine's projection back onto the mesh is the same code.

    The image and its `pix_to_face` come out of ONE rasterization, so they
    cannot disagree about which face a pixel came from. They used to come from
    two calls -- the renderer's own, and a second one made only to get the
    fragments back -- and the second passed no R and no T, relying on
    pytorch3d's cameras having STORED the pose the first call gave them. That
    worked, and it worked by accident: it is the same hidden state that makes
    two threads rendering different teeth read one tooth's mask out of the
    other's geometry.
    """
    from .torch_helpers import import_torch

    torch = import_torch()
    p3d = import_pytorch3d()
    look_at_rotation = p3d["look_at_rotation"]

    images = torch.empty(0).to(device)
    pix_to_faces = torch.empty(0).to(device)

    for direction in directions:
        camera_position = (aim + direction * radius).unsqueeze(0)
        camera_position[:, 2] -= radius * MG_CAMERA_PLUNGE
        target = aim.unsqueeze(0)
        up = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        rotation = look_at_rotation(camera_position, at=target, up=up, device=device)
        translation = -torch.bmm(rotation.transpose(1, 2), camera_position[:, :, None])[:, :, 0]

        rendered, fragments = renderer(
            meshes_world=mesh, R=rotation, T=translation.to(device)
        )
        rendered = rendered.permute(0, 3, 1, 2)[:, :-1, :, :]
        depth = fragments.zbuf.permute(0, 3, 1, 2)

        view = torch.cat([rendered, depth], dim=1)
        images = torch.cat((images, view.unsqueeze(0)), dim=0)
        pix_to_faces = torch.cat((pix_to_faces, fragments.pix_to_face.unsqueeze(0)), dim=0)

    return images.permute(1, 0, 2, 3, 4), pix_to_faces


def estimate_missing_teeth(labels, vertices, wanted, device) -> dict:
    """`{tooth number: (centre, tangent)}` for teeth absent from the labels.

    The MG cameras need a tooth centre and an arch direction, both normally read
    from the segmentation. When a tooth carries no label both can still be
    estimated: the lower ids are consecutive along the arch, so the centroids of
    the segmented teeth trace it and a quadratic fit of each coordinate against
    the id fills the gaps.

    Needs at least 4 segmented teeth spanning at least 4 ids, otherwise the
    extrapolation is not trustworthy and `{}` is returned -- the caller then
    skips those teeth, which is what happened to all of them before.
    """
    from .torch_helpers import import_torch

    torch = import_torch()

    present, centroids = [], []
    for number in wanted:
        matches = (labels[0] == int(number)).nonzero(as_tuple=True)[0]
        if len(matches):
            present.append(int(number))
            centroids.append(vertices[0][matches].mean(dim=0).detach().cpu().numpy())

    missing = [int(number) for number in wanted if int(number) not in present]
    if not missing:
        return {}
    if len(present) < 4 or (max(present) - min(present)) < 4:
        return {}

    ids = np.array(present, dtype=float)
    centroids = np.array(centroids)
    fits = [np.polyfit(ids, centroids[:, axis], deg=2) for axis in range(3)]

    def at(value):
        return np.array([np.polyval(fit, value) for fit in fits])

    return {
        number: (
            torch.tensor(at(number), dtype=torch.float32).to(device),
            torch.tensor(at(number + 0.5) - at(number - 0.5), dtype=torch.float32).to(device),
        )
        for number in missing
    }


def render_views(renderer, mesh, center, radius: float, camera_positions, device):
    """Render the tooth from every camera direction.

    Returns (images, pix_to_face): the batch the UNet consumes, and the face
    each rendered pixel came from, which is how a predicted mask gets back
    onto the mesh.

    One rasterization per view, the renderer handing back the fragments it
    shaded from. This rasterized twice -- once inside `renderer(...)` and again
    to recover `pix_to_face` -- and the second call passed neither R nor T,
    reading the pose out of the cameras the first call had left it in.

    **The mesh is handed over as it is, not cloned per view.** pytorch3d caches
    a mesh's vertex normals ON the `Meshes` object and the shader is what asks
    for them, so a fresh clone per view recomputed them per view -- 238 times
    for one scan of a maxilla, for a quantity that does not depend on the
    camera. Nothing here mutates the mesh: the rasterizer builds a new `Meshes`
    for screen space rather than moving this one, and the pose lives on the
    cameras. Measured on the reference mandible, 6 runs against 6: 0.39 s of
    4.64, and not one landmark of 69 moved.
    """
    from .torch_helpers import import_torch

    torch = import_torch()
    p3d = import_pytorch3d()
    look_at_rotation = p3d["look_at_rotation"]

    images = torch.empty(0).to(device)
    pix_to_faces = torch.empty(0).to(device)

    directions = torch.tensor(np.asarray(camera_positions, dtype=np.float32)).to(device)
    for direction in directions:
        camera_position = center + direction * radius
        rotation = look_at_rotation(camera_position, at=center, device=device)
        translation = -torch.bmm(rotation.transpose(1, 2), camera_position[:, :, None])[:, :, 0]

        rendered, fragments = renderer(
            meshes_world=mesh, R=rotation, T=translation.to(device)
        )
        rendered = rendered.permute(0, 3, 1, 2)[:, :-1, :, :]
        depth = fragments.zbuf.permute(0, 3, 1, 2)

        # The network takes 4 channels: the three normal-as-color ones plus
        # depth.
        view = torch.cat([rendered, depth], dim=1)
        images = torch.cat((images, view.unsqueeze(0)), dim=0)
        pix_to_faces = torch.cat((pix_to_faces, fragments.pix_to_face.unsqueeze(0)), dim=0)

    return images.permute(1, 0, 2, 3, 4), pix_to_faces

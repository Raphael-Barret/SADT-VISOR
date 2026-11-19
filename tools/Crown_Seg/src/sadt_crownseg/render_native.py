"""Mesh rendering without pytorch3d: Embree for the rays, torch for the rest.

On macOS the IOS engines of this repository (ALI_IOS, AREG_IOS) render a tooth
or an arch from fixed viewpoints, run a 2D UNet on the images and read back
which face each pixel came from. pytorch3d has no Metal kernel and no usable
macOS wheel for the torch the Mac needs, so on darwin this module does the
whole forward renderer instead and pytorch3d is never imported.

NOTHING HERE RUNS OFF macOS. The callers choose this module with
`sys.platform == "darwin"` and nothing else (no environment variable); on Linux
and Windows they keep pytorch3d's renderer and this file is never imported. It
imports `torch` and `numpy` at load time and `embreex` on first use, and
`embreex` is not a dependency of the package: the macOS install recipe
provides it.

What is reproduced, each part measured against pytorch3d 0.7.9 on CUDA (518 ALI
views and 7 AREG views of the public `T1_01_U_segmented.vtk`: RGB within 1e-3
on 99.9999 % of the pixels, the rest being edge pixels where both rasterizers
hit a neighbouring face):

* `look_at_rotation`, and the row-vector convention `X_view = X_world @ R + T`
  with `T = -R^T eye`, as the tools build it;
* a perspective camera with a field of view and a near plane, +X to the LEFT and
  +Y up, pixel (row r, column c) sampled at its centre
  `x = 1 - (2c + 1) / S`, `y = 1 - (2r + 1) / S`;
* the near-plane cut: pytorch3d clips the geometry at `z_view = znear / 2`, so
  the rays start ON that plane (without it 45 % of the pixels of a close view
  differ, with it 99.999 % agree);
* one face per pixel, `pix_to_face = -1` and `zbuf = -1` where nothing is hit,
  back faces not culled, barycentrics `(1 - u - v, u, v)`;
* `Meshes` reduced to one mesh: vertex normals are the sum of the unnormalized
  face normals `cross(v2 - v1, v0 - v1)`, then normalized (eps 1e-6);
* `TexturesVertex` and perspective-correct interpolation of vertex attributes;
* `HardPhongShader` with `PointLights` (pytorch3d's defaults: ambient 0.5,
  diffuse 0.3, specular 0.2, shininess 64, position (0, 1, 0)) and with
  `AmbientLights` (the colour is the interpolated texture), on a plain
  background.

Not reproduced, on purpose, because no tool here uses it: `blur_radius > 0`,
`faces_per_pixel > 1`, backface culling, batches of meshes, gradients, soft
shaders. The far plane is not applied either (pytorch3d does not cull to the
frustum by default). The pose is an ARGUMENT of every render, never state
stored in a camera object: pytorch3d's shader reads the camera centre out of
whatever pose the rasterizer last stored on ITS camera object, which is a trap
as soon as the lights have a specular term and rasterizer and shader hold two
different camera objects.

Change this file, change both copies: ALI_IOS and AREG_IOS each carry one
(scripts/tests/test_vendored_copies.py keeps them identical).
"""

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

_INSTALL_HINT = (
    "The macOS renderer needs the `embreex` package, which the macOS install "
    "recipe provides: `uv pip install embreex` in this tool's environment. It "
    "is not a dependency of the Linux deployments, which render with pytorch3d."
)


class ToolUnavailableError(RuntimeError):
    """The server maps this class by NAME to a 503 (see runner.CALLER_FACING_ERRORS)."""


def require_embree():
    """The two embreex classes, or a ToolUnavailableError saying how to install it."""
    try:
        from embreex import rtcore_scene
        from embreex.mesh_construction import TriangleMesh
    except ImportError as exc:
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: {exc.name or 'embreex'})") from exc
    return rtcore_scene, TriangleMesh


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def look_at_rotation(camera_position, at=((0.0, 0.0, 0.0),), up=((0.0, 1.0, 0.0),), device=None):
    """pytorch3d's `look_at_rotation`: (N, 3) positions -> (N, 3, 3) rotations.

    `at` and `up` are 3-element sequences or (1, 3) / (N, 3) tensors, broadcast
    against the positions. The columns of each matrix are the camera's x, y and
    z axes, so `X_view = X_world @ R + T`. When `up` is (nearly) parallel to
    the viewing direction the x axis is rebuilt from the other two, exactly as
    pytorch3d does. `device` defaults to the positions' own device (pytorch3d's
    default is the CPU).
    """
    if device is None and torch.is_tensor(camera_position):
        device = camera_position.device

    def rows(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1, 3)

    eye, at, up = torch.broadcast_tensors(rows(camera_position), rows(at), rows(up))
    z = F.normalize(at - eye, eps=1e-5)
    x = F.normalize(torch.cross(up, z, dim=1), eps=1e-5)
    y = F.normalize(torch.cross(z, x, dim=1), eps=1e-5)
    close = (x.abs() <= 5e-3).all(dim=1, keepdim=True)  # isclose(x, 0, atol=5e-3)
    if close.any():
        x = torch.where(close, F.normalize(torch.cross(y, z, dim=1), eps=1e-5), x)
    return torch.stack([x, y, z], dim=2)


def pose(eye, at=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """`(R, T)` for one camera, `R` (3, 3) and `T` (3,), with `T = -R^T eye`."""
    rotation = look_at_rotation(eye, at, up)[0]
    eye = torch.as_tensor(eye, dtype=torch.float32, device=rotation.device).reshape(3)
    return rotation, -(rotation.T @ eye)


# ---------------------------------------------------------------------------
# Mesh and textures
# ---------------------------------------------------------------------------

class Mesh:
    """One triangle mesh: `verts` (V, 3), `faces` (F, 3) int64, `colors` (V, 3).

    Everything lives on `device`; the Embree scene is built on first use and
    kept, so the second view of a mesh costs no BVH build.
    """

    def __init__(self, verts, faces, colors, device=None):
        verts = torch.as_tensor(verts, dtype=torch.float32)
        device = verts.device if device is None else torch.device(device)
        self.v = verts.to(device)
        self.f = torch.as_tensor(faces, dtype=torch.int64, device=device)
        self.c = torch.as_tensor(colors, dtype=torch.float32, device=device)
        if self.v.ndim != 2 or self.v.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3), got {tuple(self.v.shape)}")
        if self.f.ndim != 2 or self.f.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3) triangles, got {tuple(self.f.shape)}")
        if self.c.shape != self.v.shape:
            raise ValueError(f"colors must be (V, 3) like the vertices, got {tuple(self.c.shape)}")
        if self.f.numel() and (int(self.f.min()) < 0 or int(self.f.max()) >= len(self.v)):
            raise ValueError("a face names a vertex that does not exist")
        self.device = self.v.device
        self._scene = None
        self._normals = None

    def to(self, device):
        return self if torch.device(device) == self.device else Mesh(self.v, self.f, self.c, device)

    def clone(self):
        """The same mesh, caches included: nothing here mutates a mesh."""
        return self

    @property
    def normals(self):
        """Area-weighted vertex normals, pytorch3d's `verts_normals_packed`."""
        if self._normals is None:
            corners = self.v[self.f]
            face_normals = torch.cross(corners[:, 2] - corners[:, 1], corners[:, 0] - corners[:, 1], dim=1)
            total = torch.zeros_like(self.v)
            for corner in range(3):
                total = total.index_add(0, self.f[:, corner], face_normals)
            self._normals = F.normalize(total, eps=1e-6, dim=1)
        return self._normals

    @property
    def scene(self):
        if self._scene is None:
            rtcore_scene, triangle_mesh = require_embree()
            self._scene = rtcore_scene.EmbreeScene()
            triangle_mesh(
                scene=self._scene,
                vertices=np.ascontiguousarray(self.v.cpu().numpy(), np.float32),
                indices=np.ascontiguousarray(self.f.cpu().numpy(), np.int32),
            )
        return self._scene


class TexturesVertex:
    """What `pytorch3d.renderer.TexturesVertex` is here: a (1, V, 3) colour table."""

    def __init__(self, verts_features):
        self.colors = verts_features


def Meshes(verts, faces, textures):  # noqa: N802 - the name callers already use
    """A batch of ONE mesh with vertex colours, as the engines build it."""
    if verts.shape[0] != 1 or faces.shape[0] != 1:
        raise ValueError("only a batch of one mesh is supported")
    return Mesh(verts[0], faces[0], textures.colors[0])


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

_DIRECTIONS = {}


def _view_directions(fov, size):
    """Ray directions in view space with z = 1, one per pixel, row-major."""
    key = (float(fov), int(size))
    if key not in _DIRECTIONS:
        centres = 1 - (2 * np.arange(size) + 1) / size
        x, y = np.meshgrid(centres, centres)
        scale = np.tan(np.deg2rad(fov) / 2)
        _DIRECTIONS[key] = np.stack([x * scale, y * scale, np.ones_like(x)], -1).reshape(-1, 3)
    return _DIRECTIONS[key]


def rasterize(mesh, R, T, fov, znear, size):
    """`(pix_to_face (S, S) int64, zbuf (S, S), bary (S, S, 3))`, -1 where empty.

    `R` (3, 3) and `T` (3,) are the pose. Rays start on the plane
    `z_view = znear / 2`, where pytorch3d cuts the geometry, so Embree's `tfar`
    plus `znear / 2` is the view-space depth, which is pytorch3d's `zbuf`.
    """
    size = int(size)
    device = mesh.device
    # `.cpu()` BEFORE `.double()`: MPS has no float64, so the cast must happen
    # on the host, where the ray maths wants double precision anyway.
    rotation = R.detach().cpu().double().numpy()
    translation = T.detach().cpu().double().numpy()
    directions = _view_directions(fov, size) @ rotation.T
    origins = -translation @ rotation.T + (znear / 2) * directions

    if len(mesh.f) == 0:
        miss = torch.full((size, size), -1, dtype=torch.int64, device=device)
        return miss, miss.to(torch.float32), torch.zeros(size, size, 3, device=device)

    hits = mesh.scene.run(
        np.ascontiguousarray(origins, np.float32),
        np.ascontiguousarray(directions, np.float32),
        output=1,
    )
    face = np.asarray(hits["primID"]).astype(np.int64)
    hit = (face >= 0) & (np.asarray(hits["geomID"]) >= 0)
    u, v = np.asarray(hits["u"]), np.asarray(hits["v"])
    depth = np.where(hit, np.asarray(hits["tfar"]) + znear / 2, -1.0).astype(np.float32)

    pix_to_face = torch.from_numpy(np.where(hit, face, -1)).to(device).reshape(size, size)
    zbuf = torch.from_numpy(depth).to(device).reshape(size, size)
    bary = torch.from_numpy(np.stack([1 - u - v, u, v], -1).astype(np.float32)).to(device)
    return pix_to_face, zbuf, bary.reshape(size, size, 3)


def interpolate(mesh, attribute, pix_to_face, bary):
    """Barycentric interpolation of a per-vertex attribute (V, C) -> (S, S, C); 0 off the mesh."""
    if not len(mesh.f):
        return attribute.new_zeros(*pix_to_face.shape, attribute.shape[1])
    corners = attribute[mesh.f[pix_to_face.clamp(min=0)]]
    return (bary[..., None] * corners).sum(-2) * (pix_to_face >= 0)[..., None]


# ---------------------------------------------------------------------------
# Lights (the shading of `HardPhongShader`)
# ---------------------------------------------------------------------------

class PointLights:
    """pytorch3d's `PointLights()` with its default material (shininess 64)."""

    def __init__(self, ambient=0.5, diffuse=0.3, specular=0.2, location=(0.0, 1.0, 0.0), shininess=64.0):
        self.ambient, self.diffuse, self.specular = ambient, diffuse, specular
        self.location, self.shininess = location, shininess

    def shade(self, mesh, pix_to_face, bary, texels, R, T):
        points = interpolate(mesh, mesh.v, pix_to_face, bary)
        normals = F.normalize(interpolate(mesh, mesh.normals, pix_to_face, bary), dim=-1, eps=1e-6)
        light = torch.as_tensor(self.location, dtype=torch.float32, device=points.device)
        to_light = F.normalize(light - points, dim=-1, eps=1e-6)
        cosine = (normals * to_light).sum(-1)
        diffuse = self.diffuse * F.relu(cosine)[..., None]
        camera = -T.to(points.device) @ R.to(points.device).T  # camera centre, world space
        to_camera = F.normalize(camera - points, dim=-1, eps=1e-6)
        reflected = -to_light + 2 * cosine[..., None] * normals
        specular = self.specular * (F.relu((to_camera * reflected).sum(-1)) * (cosine > 0)).pow(
            self.shininess
        )[..., None]
        return (self.ambient + diffuse) * texels + specular


class AmbientLights:
    """pytorch3d's `AmbientLights()`: ambient 1, nothing else, so colour = texture."""

    def shade(self, mesh, pix_to_face, bary, texels, R, T):
        return texels


def render(mesh, R, T, lights, fov, znear, size, background=(0.0, 0.0, 0.0)):
    """`(rgb (S, S, 3), zbuf (S, S), pix_to_face (S, S))` from ONE rasterization."""
    pix_to_face, zbuf, bary = rasterize(mesh, R, T, fov, znear, size)
    texels = interpolate(mesh, mesh.c, pix_to_face, bary)
    rgb = lights.shade(mesh, pix_to_face, bary, texels, R.to(mesh.device), T.to(mesh.device))
    background = torch.as_tensor(background, dtype=torch.float32, device=mesh.device)
    return torch.where((pix_to_face < 0)[..., None], background, rgb), zbuf, pix_to_face


class Fragments(NamedTuple):
    """The three fields the engines read, shaped like pytorch3d's (1, S, S, 1)."""

    pix_to_face: torch.Tensor
    zbuf: torch.Tensor


class Renderer:
    """`MeshRendererWithFragments` for one fixed camera and one set of lights.

    Called as `renderer(meshes_world=mesh, R=R, T=T)` with `R` (1, 3, 3) and
    `T` (1, 3), it returns `(rgba (1, S, S, 4), Fragments)`: alpha is 1 where a
    face was hit and 0 elsewhere, and image and fragments come from the same
    rasterization. The pose is read from the arguments of THIS call.
    """

    def __init__(self, lights, fov, znear, size, background=(0.0, 0.0, 0.0), zfar=100.0):
        self.lights, self.fov, self.znear, self.size = lights, float(fov), float(znear), int(size)
        self.background, self.zfar = background, zfar

    def to(self, *args, **kwargs):
        return self

    def __call__(self, meshes_world, R, T, **_unused):
        rgb, zbuf, pix_to_face = render(
            meshes_world, R[0], T[0], self.lights, self.fov, self.znear, self.size, self.background
        )
        alpha = (pix_to_face >= 0).to(rgb.dtype)[..., None]
        rgba = torch.cat([rgb, alpha], dim=-1)[None]
        return rgba, Fragments(pix_to_face[None, ..., None], zbuf[None, ..., None])


def namespace():
    """The three names the engines take from pytorch3d, here, after checking Embree."""
    require_embree()
    return {"Meshes": Meshes, "TexturesVertex": TexturesVertex, "look_at_rotation": look_at_rotation}

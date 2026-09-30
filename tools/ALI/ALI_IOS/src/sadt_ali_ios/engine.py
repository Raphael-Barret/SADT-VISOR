"""The IOS landmark pipeline: render each tooth, predict masks, project back.

Ported from ALI_IOS/ALI_IOS.py. Per tooth, the mesh is rendered from a dozen
viewpoints, a 2D UNet predicts one mask channel per landmark type, and each
mask is projected back onto the mesh faces that produced its pixels; the
landmark is the surface point nearest the centroid of those faces.

Restructured around what is loaded when. The original built a renderer,
instantiated a UNet and called `load_state_dict` INSIDE the per-tooth loop --
28 model loads per scan per network, re-reading the mesh every time. Here the
mesh is read once per scan, the renderer once per run, and the weights once per
(network, jaw). The inference itself is unchanged.

Three defects fixed, all of which cost results silently:

* a jaw whose weights are missing raised a `KeyError` that was caught and
  discarded, so the jaw vanished from the output. It is reported now.
* a mesh with no tooth labels fell back to all-zeros, so no tooth was ever
  found and the run ended with no landmarks and no reason. Every mesh is
  checked for labels up front now, and a batch missing them names Crown_Seg.
* `.stl` input was accepted by the UI and then never discovered by the CLI,
  which globbed for `.vtk` only.
"""

import logging
import os
import time

import numpy as np

from .torch_helpers import import_torch, resolve_device
from .errors import ToolInputError, ToolUnavailableError
from sadt_ali_common.markups import MARKUPS_EXTENSION
from sadt_ali_common.markups import write as write_markups
from . import catalog
from . import progress
from . import render, surface

logger = logging.getLogger(__name__)

_MONAI_HINT = "ALI's IOS engine needs monai. Run `uv sync` in tools/ALI."


def _import_unet():
    try:
        from monai.networks.nets import UNet
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_MONAI_HINT} (missing: monai)") from exc
    return UNet


def check_dependencies() -> None:
    """Import the whole lazy stack once, before any mesh is touched.

    Same reason as the CBCT engine's: a missing dependency belongs to the
    venv, not to a patient's mesh, and the per-mesh `except` below would
    otherwise report it once per mesh and then hide it behind "produced no
    landmarks for any mesh". It matters more here -- pytorch3d is an optional
    extra compiled from source, so this is the failure an IOS run actually hits
    on a venv synced without it, and it has to say so in one clear line.
    """
    import_torch()
    surface.import_vtk()
    render.import_pytorch3d()
    _import_unet()


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def discover_weights(model_path: str):
    """({network: {jaw: path}}, unrecognized file names) for an IOS bundle.

    **One naming rule, stated once.** A checkpoint's base name is split on
    `_`; it must contain a token naming the network (`O` or `C`) and a token
    naming the jaw (`Upper` or `Lower`). The published weights are called
    `Upper_O_model.pth`, `Lower_C_model.pth` and so on, so both rules the
    original used agreed on them -- but they were different rules (the UI
    required the substring `_O_`, the CLI took `basename.split("_")[1]`), and
    a bundle named slightly differently was accepted by one and rejected by
    the other.

    The jaw must be named explicitly. The original treated every file not
    containing "Lower" as upper-jaw weights, so a bundle missing its mandibular
    model quietly predicted the lower arch with the maxillary one.
    """
    if not os.path.isdir(model_path):
        # An argument error, not a fault of this tool: the caller pointed
        # `model` at something that is not an IOS bundle. Basename only -- the
        # message reaches the client verbatim, the server's paths do not.
        raise ToolInputError(
            f"IOS model bundle '{os.path.basename(model_path.rstrip(os.sep))}' "
            f"is not a directory."
        )

    weights: dict = {}
    unrecognized = []
    # Where each (network, jaw) was found -- see the collision check below.
    seen: dict = {}

    for root, _dirs, files in os.walk(model_path):
        for name in sorted(files):
            if not name.endswith(".pth"):
                continue
            tokens = {token.lower() for token in os.path.splitext(name)[0].split("_")}
            network = next((code for code in catalog.NETWORK_CODES if code.lower() in tokens), None)
            jaw = next((candidate for candidate in catalog.JAWS if candidate.lower() in tokens), None)
            if network is None or jaw is None:
                unrecognized.append(name)
                continue
            where = os.path.relpath(root, model_path)
            if (network, jaw) in seen:
                # Two bundles under one folder. The assignment below used to let
                # the second silently overwrite the first, so a run could mix two
                # vintages with nothing in the report to say which. Relative
                # paths: this message reaches the client verbatim.
                raise ToolInputError(
                    "This model folder holds more than one bundle: {} weights for "
                    "the {} jaw are in both '{}' and '{}'. Point 'model' at one of "
                    "them -- choosing here would leave which weights ran "
                    "unrecorded.".format(network, jaw.lower(), seen[(network, jaw)], where)
                )
            seen[(network, jaw)] = where
            weights.setdefault(network, {})[jaw] = os.path.join(root, name)

    return weights, sorted(unrecognized)


def require_labels(meshes: list) -> None:
    """Refuse a batch whose meshes carry no tooth labels, naming what makes them.

    This is where `ALILogic.ensure_segmented()` used to call CrownSeg in-process
    and hand the labelled mesh straight on. Tools no longer call each other, so
    the chain is the server's to run -- and the tool's job is to say so in a way
    the person who sent the request can act on, rather than failing per mesh
    with "no known tooth number is present".

    Checked for the WHOLE batch before any weights are loaded: reading a mesh's
    point-data array names is cheap, and discovering this on mesh 40 of 40 after
    an hour of inference is the failure worth spending that scan on.
    """
    unlabelled = [key for path, key in meshes if surface.label_array_name(
        surface.read_surface(path)) is None]
    if not unlabelled:
        return
    raise ToolInputError(
        f"{len(unlabelled)} of {len(meshes)} mesh(es) carry no tooth labels, so there is "
        f"nothing for the landmark networks to be pointed at. Expected a point-data array "
        f"named one of: {', '.join(surface.LABEL_ARRAY_NAMES)}. Run 'Crown_Seg' over these "
        f"meshes first and send its output here -- its run report lists every labelled mesh "
        f"under 'segmented_meshes'."
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_network(checkpoint: str, device: str, network_code: str = "O"):
    """The 2D UNet the IOS weights were trained as.

    Every argument here is part of the checkpoint's shape; changing one makes
    `load_state_dict` fail rather than degrade, which is why the two shapes are
    declared rather than inferred.

    The mucogingival network is a different shape. O and C take one image per
    camera as a batch: 4 channels in (normals as RGB + depth), 4 classes out.
    MG stacks its three buccal views into ONE input of 12 channels and predicts
    3 classes -- one landmark to find, needing the three views together.
    """
    torch = import_torch()
    UNet = _import_unet()

    in_channels, out_channels = (12, 3) if network_code == "MG" else (4, 4)
    network = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=4,
    ).to(device)
    network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    network.eval()
    return network


def _predicted_faces(predictions, pix_to_face, channel: int) -> list:
    """Face indices whose rendered pixels were predicted as `channel`.

    `predictions` is the UNet output, (views, channels, H, W); `pix_to_face`
    says which mesh face each rendered pixel came from, shaped
    (views, 1, H, W, faces_per_pixel=1). A pixel where nothing was rendered
    carries -1, and those are dropped -- the original passed them on, where
    `faces[-1]` silently selected the mesh's last face and dragged the
    landmark's centroid to an arbitrary corner of the arch.

    The class is taken by argmax over the raw logits. The original cast them
    to int16 first, truncating every value toward zero, which turned near ties
    into exact ties that argmax then resolved in favour of the background
    channel -- shrinking every mask for no reason anyone intended.

    **One gather, not one device round trip per predicted pixel.** The loop
    this replaces called `.item()` on a card-resident tensor once per pixel the
    mask claimed, and every one of those is a synchronisation: 3.46 s of a
    21.6 s run (16.0 %) to read 119,456 integers that one indexing operation
    reads at once. `nonzero` already returns its hits in row-major
    (view, row, column) order, which is the order the loop visited them in, so
    the list is identical element for element -- verified against the loop on
    all 70 calls of a real run. That order is load-bearing: the caller sums
    these faces' vertices sequentially in float32.
    """
    torch = import_torch()

    classes = torch.argmax(predictions, dim=1)
    hits = (classes == channel).nonzero(as_tuple=False)
    if hits.numel() == 0:
        return []
    # `pix_to_face` is float32 here, not int64: it is accumulated into a tensor
    # `torch.empty(0)` created, and cat promotes. `.long()` truncates toward
    # zero exactly as the `int(...)` it replaces did, and a mesh would need
    # 16.7M faces before float32 could not hold an index exactly.
    picked = pix_to_face[hits[:, 0], 0, hits[:, 1], hits[:, 2], 0].long()
    # A pixel where nothing was rendered carries -1 and is dropped.
    return picked[picked >= 0].tolist()


def _landmark_position(faces, face_table, vertices, locator, scaled_surface):
    """The surface point nearest the centroid of a set of faces.

    Snapped onto an actual mesh point rather than left as the raw centroid,
    which would float inside the tooth: a landmark is a point *on* the crown.

    **The centroid is computed to the last bit of what the loop produced**, and
    that constraint is what shapes this function. What it replaces read three
    card-resident integers per face with `.item()`, then added the vertices one
    tensor at a time -- so a mask of 1,622 faces cost 4,866 device
    synchronisations and 4,866 kernel launches to add 4,866 rows of three
    floats. Measured: 5.17 s of a 21.6 s run, the single most expensive thing
    in the engine, and 24 % of it, against 1.6 % for the inference it exists to
    interpret.

    Two details make the vectorised form bit-identical rather than merely
    close, and neither was arrived at by reasoning -- both are what measurement
    forced, after a first version that looked obviously correct was not:

    * **a sequential accumulation, and numpy's**, not `sum(dim=0)`. float32
      addition is not associative, so a tree reduction is a different number
      from adding the rows one after another -- about a ULP per thousand rows.
      `torch.cumsum` is sequential on CUDA and NOT on the CPU, which this tool
      also runs on, so it is right on the card and wrong in the test suite;
      `np.add.accumulate`, which `np.cumsum` is, is sequential on both.
      Verified against the loop at five sizes from 3 to 14,220 rows and on
      every call of a real run, on both devices.
    * **the division stays in torch, on the tensor's own device.** Moved to
      numpy it came out one ULP different, because torch divides a tensor by a
      scalar by multiplying by the reciprocal. That is a 6e-08 shift in
      unit-sphere space -- 2 nanometres of patient, which would never have
      moved the snapped point, and would still have made this function
      something other than what it replaced.

    So the total comes back to the device as three floats to be divided there.
    Two 12-byte transfers, against the 4,866 synchronisations they replace.
    """
    torch = import_torch()

    if not len(faces):
        return None
    wanted = torch.as_tensor(faces, dtype=torch.long, device=face_table.device)
    # Face-major, corner-minor: the order the nested loop visited them in.
    vertex_ids = face_table[0][wanted].reshape(-1)
    gathered = vertices[0][vertex_ids]

    total = np.cumsum(gathered.detach().cpu().numpy(), axis=0, dtype=np.float32)[-1]
    centroid = torch.from_numpy(total).to(gathered.device) / vertex_ids.numel()
    point_id = locator.FindClosestPoint(centroid.detach().cpu().numpy())
    return scaled_surface.GetPoint(point_id)


def _predict_one_scan(mesh_path, key, record, weights, networks, device, renderer,
                      output_dir, prediction_ID,
                      mesh_index: int = 1, mesh_total: int = 1) -> None:
    """Every requested network, on every tooth this mesh actually carries.

    Logs one line per (network, jaw) pass. Teeth are rendered and predicted one
    at a time, so a full arch is dozens of GPU passes -- without a line in
    between, a run looks like a hang. Counts and anatomical labels only, never
    the mesh's file name.
    """
    torch = import_torch()
    p3d = render.import_pytorch3d()
    vtk, _ = surface.import_vtk()

    raw = surface.read_surface(mesh_path)
    scaled, mesh_center, scale_factor = surface.scale_to_unit(raw)
    vertices, faces, colors, labels = surface.surface_properties(scaled, device)

    mesh = p3d["Meshes"](
        verts=vertices, faces=faces, textures=p3d["TexturesVertex"](verts_features=colors)
    ).to(device)

    locator = vtk.vtkOctreePointLocator()
    locator.SetDataSet(scaled)
    locator.BuildLocator()

    present = {int(value) for value in labels.squeeze(0).unique().tolist()}
    teeth_by_jaw = {jaw: [] for jaw in catalog.JAWS}
    for jaw, numbers in catalog.UNIVERSAL_NUMBERS.items():
        teeth_by_jaw[jaw] = sorted(number for number in numbers.values() if number in present)

    if not any(teeth_by_jaw.values()):
        raise RuntimeError(
            "no known tooth number is present in this mesh's label array "
            "(expected Universal numbering, 2-15 upper and 18-31 lower)"
        )

    positions = {}
    notes: dict = {}
    for network in networks:
        for jaw, teeth in teeth_by_jaw.items():
            # MG was trained on the mandible alone, so a maxilla is not a
            # missing model -- it is a question the network cannot be asked.
            if jaw not in catalog.NETWORK_JAWS.get(network, catalog.JAWS):
                continue
            if network == "MG":
                # Every MG tooth, not only the segmented ones: the ones with no
                # label get their cameras aimed from a fit of the arch through
                # the ones that have (see render.estimate_missing_teeth), which
                # is what keeps a gap in the segmentation from becoming a gap in
                # the mucogingival line.
                teeth = list(catalog.MG_TEETH)
            if not teeth:
                continue
            checkpoint = weights.get(network, {}).get(jaw)
            if checkpoint is None:
                # Reported, not swallowed: the original's KeyError here was
                # caught and discarded, so the jaw silently disappeared.
                record["jaws_without_model"].setdefault(
                    catalog.NETWORK_DISPLAY_NAMES.get(network, network), []
                ).append(jaw)
                continue

            # Loaded once per (network, jaw), not once per tooth.
            pass_started = time.monotonic()
            pass_name = f"{catalog.NETWORK_DISPLAY_NAMES.get(network, network)}/{jaw}"
            before = len(record["landmarks_found"])
            logger.info(
                "mesh %d/%d: %s -- %d tooth/teeth",
                mesh_index, mesh_total, pass_name, len(teeth),
            )

            unet = _build_network(checkpoint, device, network)
            estimates = (
                render.estimate_missing_teeth(labels, vertices, teeth, device)
                if network == "MG"
                else {}
            )
            if estimates:
                logger.info(
                    "mesh %d/%d: %s -- %d tooth/teeth absent from the segmentation, "
                    "positions estimated from the arch",
                    mesh_index, mesh_total, pass_name, len(estimates),
                )
            try:
                for tooth_number in teeth:
                    label_names = catalog.LABELS[network].get(str(tooth_number))
                    if label_names is None:
                        continue
                    try:
                        found = _predict_one_tooth(
                            unet=unet,
                            renderer=renderer,
                            mesh=mesh,
                            network=network,
                            jaw=jaw,
                            tooth_number=tooth_number,
                            label_names=label_names,
                            vertices=vertices,
                            faces=faces,
                            labels=labels,
                            locator=locator,
                            scaled=scaled,
                            mesh_center=mesh_center,
                            scale_factor=scale_factor,
                            device=device,
                            estimated=estimates.get(tooth_number),
                            notes=notes,
                        )
                    except Exception as exc:
                        # One tooth failing must not cost the other 27.
                        logger.exception("IOS prediction raised for tooth %d", tooth_number)
                        record["landmarks_failed"][f"{jaw}-{tooth_number}"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue

                    positions.update(found)
                    record["landmarks_found"].extend(sorted(found))
            finally:
                del unet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(
                    "mesh %d/%d: %s -- %d landmark(s) in %.0fs",
                    mesh_index, mesh_total, pass_name,
                    len(record["landmarks_found"]) - before,
                    time.monotonic() - pass_started,
                )

    if not positions:
        raise RuntimeError("no landmark was predicted on this mesh")

    record["landmarks_found"].sort()
    if notes:
        # In the report AND in the file (see write_markups): a point placed from
        # an arch fit or forced out of the most likely pixels is a point to
        # review, and it is indistinguishable from a good one otherwise.
        record["landmarks_degraded"] = dict(sorted(notes.items()))
    destination = os.path.join(
        output_dir,
        os.path.dirname(key),
        f"{os.path.splitext(os.path.basename(mesh_path))[0]}_lm_{prediction_ID}"
        f"{MARKUPS_EXTENSION}",
    )
    record["produced"].append(write_markups(positions, destination, descriptions=notes))


def _predict_one_tooth(unet, renderer, mesh, network, jaw, tooth_number, label_names,
                       vertices, faces, labels, locator, scaled, mesh_center, scale_factor,
                       device, estimated=None, notes=None) -> dict:
    """Render one tooth, predict its landmarks, return {label: position}."""
    torch = import_torch()

    if network == "MG":
        return _predict_mucogingival(
            unet=unet, renderer=renderer, mesh=mesh, tooth_number=tooth_number,
            label_name=label_names[0], vertices=vertices, faces=faces, labels=labels,
            locator=locator, scaled=scaled, mesh_center=mesh_center,
            scale_factor=scale_factor, device=device, estimated=estimated, notes=notes,
        )

    center = render.tooth_center(labels, vertices, tooth_number, device)
    if center is None:
        return {}

    images, pix_to_face = render.render_views(
        renderer=renderer,
        mesh=mesh,
        center=center,
        radius=catalog.CAMERA_RADIUS[network],
        camera_positions=render.CAMERA_POSITIONS[network][jaw],
        device=device,
    )
    # (1, views, 4, H, W) -> (views, 4, H, W): one image per camera, which is
    # the batch the UNet consumes.
    with torch.no_grad():
        predictions = unet(images[0].float().to(device))

    found = {}
    # Channel 0 of the network output is background; the landmark types start
    # at 1, in the order catalog.NETWORKS declares them.
    for channel, label_name in enumerate(label_names, start=1):
        predicted = _predicted_faces(predictions, pix_to_face, channel)
        kept = surface.faces_on_tooth(faces, predicted, labels, tooth_number)
        position = _landmark_position(kept, faces, vertices, locator, scaled)
        if position is not None:
            found[label_name] = surface.upscale(position, mesh_center, scale_factor)
    return found


MG_FORCE_TOPK = 50


def _predict_mucogingival(unet, renderer, mesh, tooth_number, label_name, vertices, faces,
                          labels, locator, scaled, mesh_center, scale_factor, device,
                          estimated=None, notes=None) -> dict:
    """One mucogingival landmark, from the three adaptive buccal views.

    Three things differ from the crown networks, and each is why the naive
    version of this returned nothing:

    * the cameras are aimed PER TOOTH, along the buccal normal of the arch at
      that tooth (see `render.mg_frame`);
    * the predicted faces are NOT filtered to the tooth. The mucogingival point
      is on the gingiva, and `faces_on_tooth` keeps only faces whose vertices
      carry this tooth's label -- it would drop every one of them;
    * a tooth that predicts no pixel at all falls back to its most likely ones
      rather than being dropped, a gap in the mucogingival line being what AREG
      then has to fit a spline through.
    """
    torch = import_torch()

    if estimated is not None:
        center, tangent = estimated
        center = center.view(1, 3)
        tangent = tangent.clone()
        tangent[2] = 0.0
        norm = torch.norm(tangent)
        tangent = tangent / norm if norm > 1e-6 else None
    else:
        center = render.tooth_center(labels, vertices, tooth_number, device)
        if center is None:
            return {}
        tangent = render.arch_tangent(labels, vertices, tooth_number, device)

    normal, aim = render.mg_frame(vertices, center, tangent, tooth_number, device)

    # The GPU semaphore went with the split: each tool is its own process now,
    # so an in-process one serialised nothing. Its definition was removed and
    # this use site survived -- a NameError on the mucogingival path, which had
    # never run until AREG_IOSCBCT called it.
    images, pix_to_face = render.render_mg_views(
        renderer=renderer,
        mesh=mesh,
        aim=aim,
        directions=render.mg_camera_directions(normal, device),
        radius=catalog.CAMERA_RADIUS["MG"],
        device=device,
    )
    # The three views become the 12 channels of ONE input, not a batch of
    # three: (1, 3, 4, H, W) -> (1, 12, H, W).
    views = images[0].unsqueeze(0)
    batch, cameras, channels, height, width = views.shape
    with torch.no_grad():
        predictions = unet(
            views.reshape(batch, cameras * channels, height, width).float().to(device)
        )

    # argmax on the raw scores. Casting the logits to int16 first -- what the
    # crown path inherited and this deliberately does not -- truncates every
    # value toward zero, turning near ties into exact ties that argmax then
    # resolves in favour of channel 0, the background.
    logits = predictions.detach().float()
    classes = torch.argmax(logits, dim=1)
    chosen = (classes == 1).nonzero(as_tuple=False)

    note = None
    if len(chosen) == 0:
        # Nothing won. Keep the pixels where the landmark class is most likely
        # anyway: upstream measures a forced point at ~5 mm of error against
        # ~1.2 mm for a won one, which is worth having as a point to be reviewed
        # rather than a hole in the line.
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        keep = min(MG_FORCE_TOPK, probabilities.numel())
        if keep == 0:
            return {}
        confidence, flat = probabilities.reshape(-1).topk(keep)
        chosen = torch.stack(
            torch.unravel_index(flat, probabilities.shape), dim=1
        )
        note = f"forced (confidence {float(confidence.max()):.3f})"

    # `chosen` indexes (view, row, column) of the stacked prediction; the same
    # (view, row, column) of pix_to_face says which face was rendered there.
    predicted = []
    for entry in chosen:
        view, row, column = (int(value) for value in entry.tolist())
        face = int(pix_to_face[view, 0, row, column, 0].item())
        if face >= 0:
            predicted.append(face)

    position = _landmark_position(predicted, faces, vertices, locator, scaled)
    if position is None:
        return {}

    if estimated is not None:
        note = "; ".join(
            filter(None, [note, "cameras aimed from an arch fit, tooth not segmented"])
        )
    if note and notes is not None:
        # Recorded rather than logged: a degraded point looks exactly like a
        # good one in the output file, and whoever reads the landmarks is the
        # one who needs to know which ones to review.
        notes[label_name] = note
    return {label_name: surface.upscale(position, mesh_center, scale_factor)}


def predict_landmarks(
    meshes: list,
    model_path: str,
    networks=None,
    prediction_ID: str = "Pred",
    output_dir: str = None,
    device: str = None,
) -> dict:
    """Place landmarks on every mesh; return the run report.

    `meshes` is a list of `(absolute path, key)` pairs, the key being the
    path relative to the input root -- so a batch keeps its tree and two
    patients named `scan.vtk` in different folders cannot overwrite each other.
    Every mesh must already carry tooth labels: `require_labels` below refuses
    the batch otherwise, naming the tool that produces them.
    """
    started_at = time.monotonic()

    check_dependencies()
    device = resolve_device(device)
    networks = tuple(networks) if networks is not None else catalog.NETWORK_CODES
    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    weights, unrecognized = discover_weights(model_path)
    available = [network for network in networks if network in weights]
    if not available:
        network_names = ", ".join(
            catalog.NETWORK_DISPLAY_NAMES.get(code, code) for code in networks
        )
        # An input error, not a crash: nothing the server can do -- the caller
        # must pick the bundle (or the networks) that match. Slicer shows this
        # message verbatim.
        raise ToolInputError(
            f"'{os.path.basename(model_path)}' has no IOS weights for the selected network(s) "
            f"({network_names}). Checkpoints must be named with an 'O' or 'C' token and an "
            f"'Upper' or 'Lower' token, e.g. Upper_O_model.pth."
            + (f" Unrecognized: {', '.join(unrecognized)}." if unrecognized else "")
        )

    require_labels(meshes)

    renderer = render.build_renderer(device)
    logger.info(
        "ALI IOS: %d mesh(es), %d network(s), device=%s", len(meshes), len(available), device
    )

    scan_reports = {}
    for mesh_index, (mesh_path, key) in enumerate(meshes, start=1):
        record = {
            "input": os.path.basename(mesh_path),
            "status": "pending",
            "landmarks_found": [],
            "landmarks_failed": {},
            "jaws_without_model": {},
            "produced": [],
        }
        scan_reports[key] = record
        scan_started = time.monotonic()
        # Position in the batch, never the mesh's name -- see the CBCT engine.
        # The progress event carries the same counter, for the same reason.
        logger.info("mesh %d/%d: reading and scaling", mesh_index, len(meshes))
        progress.report(mesh_index, len(meshes), "mesh")

        try:
            _predict_one_scan(
                mesh_path=mesh_path,
                key=key,
                record=record,
                weights=weights,
                networks=available,
                device=device,
                renderer=renderer,
                output_dir=output_dir,
                prediction_ID=prediction_ID,
                mesh_index=mesh_index,
                mesh_total=len(meshes),
            )
            record["status"] = "ok"
        except Exception as exc:
            logger.exception("ALI IOS failed on one mesh")
            record["status"] = "failed"
            record["error"] = str(exc)
        record["duration_seconds"] = round(time.monotonic() - scan_started, 2)
        logger.info(
            "mesh %d/%d: %s -- %d landmark(s), %.0fs",
            mesh_index, len(meshes), record["status"],
            len(record["landmarks_found"]), record["duration_seconds"],
        )

    written = sum(len(record["landmarks_found"]) for record in scan_reports.values())
    logger.info(
        "ALI IOS done: %d/%d mesh(es), %d landmark(s) written, %.0fs",
        sum(1 for r in scan_reports.values() if r["status"] == "ok"),
        len(scan_reports), written, time.monotonic() - started_at,
    )

    processed = [record for record in scan_reports.values() if record["status"] == "ok"]
    if not processed:
        first_error = next(
            (record.get("error") for record in scan_reports.values() if record.get("error")),
            "unknown",
        )
        raise RuntimeError(f"ALI produced no landmarks for any mesh. First error: {first_error}")

    return {
        "mode": "IOS",
        "device": device,
        # The bundle the weights CAME FROM, not the argument that was passed.
        # The server hands a hosted-model argument the whole of
        # `DATA/<tool>/models/` when the caller names no bundle, so the
        # argument's own basename is "models" -- which is exactly what this
        # field exists not to say. Taken from a checkpoint that was actually
        # loaded, so it names the weights that ran even when nobody chose them.
        "model_bundle": os.path.basename(os.path.dirname(next(
            path for jaws in weights.values() for path in jaws.values()))),
        "prediction_ID": prediction_ID,
        "networks": [catalog.NETWORK_DISPLAY_NAMES.get(code, code) for code in available],
        # Same meaning as the CBCT engine's list, and read by the same client
        # code: a network requested but absent from the bundle.
        "landmarks_without_model": [
            catalog.NETWORK_DISPLAY_NAMES.get(code, code)
            for code in networks
            if code not in weights
        ],
        "models_unrecognized": unrecognized,
        "cases": scan_reports,
        "summary": {
            "total": len(scan_reports),
            "processed": len(processed),
            "failed": len(scan_reports) - len(processed),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }

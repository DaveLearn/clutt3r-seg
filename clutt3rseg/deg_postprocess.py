"""Shared deg-harness scene filters, ported for parity with the other baselines.

The deg-adapted baselines (SAM3D, MaskClustering, Open3DIS, SAI3D) all finish
``initialize_scene`` with the same three scene-level filters, driven by the deg
:class:`SceneSetup` (``ground_gaussians`` + ``ground_plane``):

1. **workspace crop** - keep only points inside a voxel grid extruded from the
   table plane (1 m up, 0.1 m below, eroded inwards in XY); drop instances that
   are mostly outside it.
2. **min-frame >= 3** - drop instances observed in fewer than 3 views (the
   caller relaxes this to >=2 for <=3-view setups, matching the other baselines).
3. **table removal** - drop the single instance lying on the table plane.

Clutt3R-Seg natively does none of these - it exports every recovered instance -
so this module adds them for a fair comparison. Everything operates in 3D on the
exported per-instance point clouds (``inst2all_points``: ``{inst_id: (N, 6)}``,
xyz+rgb) plus the leaf->instance map (``node2inst``: ``{(frame_idx, leaf_id):
inst_id}``), which together are the natural analogue of the mask/point filters
the other baselines run.

The numeric constants are copied verbatim from
``SegmentAnything3D/segmenter.py`` (``get_workspace_voxels``,
``_erode_voxel_grid_xy``) and ``SegmentAnything3D/util.py``
(``determine_table_instance_id``) so the filters behave identically across
baselines.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

logger = logging.getLogger("clutt3rseg-degfilter")

# --- constants shared with the other deg baselines -------------------------------
WORKSPACE_VOXEL_SIZE = 0.02
WORKSPACE_DESIRED_HEIGHT = 1.0
WORKSPACE_BELOW_TABLE_HEIGHT = 0.10
WORKSPACE_SHRINK_XY_M = 0.04
MIN_WORKSPACE_INSIDE_FRAC = 0.5  # SAM3D get_pcd drops a mask below 0.5 inside-workspace
TABLE_PLANE_DIST = 0.02  # determine_table_instance_id: |signed dist| < 0.02 m
TABLE_FRAC_THRESH = 0.7  # ... and > 70% of the instance lies on the plane
MIN_FRAME_COUNT = 3


def _erode_voxel_grid_xy(voxel_grid: o3d.geometry.VoxelGrid, layers: int) -> o3d.geometry.VoxelGrid:
    """Shrink the workspace footprint inwards by ``layers`` voxels in XY (verbatim
    from SegmentAnything3D/segmenter.py)."""
    if layers <= 0 or not voxel_grid.has_voxels():
        return voxel_grid

    voxel_indices = [tuple(int(idx) for idx in voxel.grid_index) for voxel in voxel_grid.get_voxels()]
    xy_occupied = {(x, y) for x, y, _ in voxel_indices}

    for _ in range(layers):
        if not xy_occupied:
            break
        prev_xy = xy_occupied
        xy_occupied = {
            (x, y) for (x, y) in prev_xy if ((x - 1, y) in prev_xy and (x + 1, y) in prev_xy and (x, y - 1) in prev_xy and (x, y + 1) in prev_xy)
        }

    for voxel_index in voxel_indices:
        if (voxel_index[0], voxel_index[1]) not in xy_occupied:
            voxel_grid.remove_voxel(voxel_index)

    return voxel_grid


def build_workspace_voxels(scene, voxel_size: float = WORKSPACE_VOXEL_SIZE, shrink_xy_m: float = WORKSPACE_SHRINK_XY_M) -> o3d.geometry.VoxelGrid:
    """Voxel grid extruded from the table plane, matching the other baselines'
    ``get_workspace_voxels``. ``scene`` is a deg ``SceneSetup``."""
    table_xyz = np.asarray(scene.ground_gaussians.xyz)
    table_plane = scene.ground_plane
    table_normal = np.array([table_plane[0], table_plane[1], table_plane[2]])
    extruded = table_xyz.copy()

    iters = int(np.ceil(WORKSPACE_DESIRED_HEIGHT / voxel_size))
    for i in range(iters):
        extruded = np.append(extruded, table_xyz + table_normal * voxel_size * i, axis=0)

    below_iters = int(np.ceil(WORKSPACE_BELOW_TABLE_HEIGHT / voxel_size))
    for i in range(below_iters):
        extruded = np.append(extruded, table_xyz - table_normal * voxel_size * (i + 1), axis=0)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(extruded))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    layers = max(0, int(np.round(shrink_xy_m / voxel_grid.voxel_size)))
    return _erode_voxel_grid_xy(voxel_grid, layers)


def frame_counts_from_node2inst(node2inst: dict) -> dict[int, int]:
    """Number of distinct frames each instance draws a leaf mask from."""
    inst2frames: dict[int, set] = defaultdict(set)
    for (fidx, _lid), inst_id in node2inst.items():
        inst2frames[int(inst_id)].add(int(fidx))
    return {inst_id: len(frames) for inst_id, frames in inst2frames.items()}


def filter_by_frame_count(inst2pts: dict, node2inst: dict, min_frames: int = MIN_FRAME_COUNT) -> dict:
    """Drop instances observed in fewer than ``min_frames`` views (cf. the other
    baselines' ">= 3 frames" rule)."""
    counts = frame_counts_from_node2inst(node2inst)
    kept = {}
    for inst_id, arr in inst2pts.items():
        c = counts.get(int(inst_id), 0)
        if c < min_frames:
            logger.info("min-frame: dropping instance %s (seen in %d < %d frames)", inst_id, c, min_frames)
            continue
        kept[inst_id] = arr
    return kept


def crop_instances_to_workspace(inst2pts: dict, workspace_voxels: o3d.geometry.VoxelGrid, min_inside_frac: float = MIN_WORKSPACE_INSIDE_FRAC) -> dict:
    """Keep only the in-workspace points of each instance; drop instances with less
    than ``min_inside_frac`` of their points inside the workspace."""
    kept = {}
    for inst_id, arr in inst2pts.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        pts = arr[:, :3].astype(np.float64)
        inside = np.asarray(workspace_voxels.check_if_included(o3d.utility.Vector3dVector(pts)), dtype=bool)
        n_in = int(inside.sum())
        frac = n_in / len(arr)
        if n_in == 0 or frac < min_inside_frac:
            logger.info("workspace: dropping instance %s (%d/%d = %.0f%% inside)", inst_id, n_in, len(arr), 100 * frac)
            continue
        kept[inst_id] = arr[inside]
    return kept


def determine_table_instance(inst2pts: dict, ground_plane, dist_thresh: float = TABLE_PLANE_DIST, frac_thresh: float = TABLE_FRAC_THRESH) -> int:
    """Return the id of the instance lying on the table plane, or -1.

    3D analogue of ``determine_table_instance_id``: among instances with more than
    ``frac_thresh`` of their points within ``dist_thresh`` of the plane, pick the
    one with the most near-plane points (the table is the largest such surface)."""
    a, b, c, d = ground_plane
    norm = math.sqrt(a * a + b * b + c * c) or 1.0
    best_id, best_count = -1, -1
    for inst_id, arr in inst2pts.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        pts = arr[:, :3]
        dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / norm
        n_near = int((dist < dist_thresh).sum())
        if n_near / len(arr) > frac_thresh and n_near > best_count:
            best_count, best_id = n_near, int(inst_id)
    return best_id


def build_instance_masks_from_leaves(
    node2inst: dict,
    kept_inst_ids,
    mask_dir,
    frame_specs,
) -> tuple[list[int], list[np.ndarray]]:
    """Per-frame instance-id masks (the form the deg harness requires), built by
    painting each surviving instance's Grounded-SAM leaf masks.

    This is clutt3r-seg's canonical 2D labeling: ``node2inst`` assigns every leaf
    mask to exactly one instance, so painting a leaf with its instance id is exact
    (it mirrors MaskClustering's ``_build_instance_groups_from_clustered_masks``,
    which paints per-frame source masks by object). Object ids are 1-indexed,
    0 = background; only ``kept_inst_ids`` (post-filter survivors) are painted.

    ``frame_specs`` is ``[(workspace_frame_idx, deg_frame_id, h, w), ...]`` for
    *all* deg frames, so the harness finds a mask for every ``frame.id``; frames
    that contributed no surviving leaf get an all-zero mask. Returns
    ``(frame_ids, pixel_object_ids)``.
    """
    inst2out = {int(inst_id): k + 1 for k, inst_id in enumerate(sorted(kept_inst_ids))}
    frame_leaves: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (fidx, lid), inst_id in node2inst.items():
        out_id = inst2out.get(int(inst_id))
        if out_id is not None:
            frame_leaves[int(fidx)].append((int(lid), out_id))

    frame_ids: list[int] = []
    pixel_object_ids: list[np.ndarray] = []
    for fidx, deg_id, h, w in frame_specs:
        canvas = np.zeros((h, w), np.int32)
        painted: list[tuple[int, np.ndarray, int]] = []
        for lid, out_id in frame_leaves.get(fidx, []):
            mpath = Path(mask_dir) / f"mask_{fidx:06d}_{lid:02d}.png"
            if not mpath.exists():
                continue
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            painted.append((int((m > 0).sum()), m > 0, out_id))
        # Paint larger leaves first so a smaller, more specific mask wins on overlap.
        for _, m, out_id in sorted(painted, key=lambda t: -t[0]):
            canvas[m] = out_id
        frame_ids.append(int(deg_id))
        pixel_object_ids.append(canvas)
    return frame_ids, pixel_object_ids


def apply_deg_filters(
    inst2pts: dict,
    node2inst: dict,
    scene,
    *,
    min_frames: int = MIN_FRAME_COUNT,
    workspace_voxel_size: float = WORKSPACE_VOXEL_SIZE,
    remove_table: bool = True,
) -> tuple[dict, int]:
    """Apply the three shared deg-harness filters in the baselines' order
    (workspace crop -> min-frame -> table removal). Returns ``(filtered, table_id)``."""
    n0 = len(inst2pts)
    workspace_voxels = build_workspace_voxels(scene, voxel_size=workspace_voxel_size)
    out = crop_instances_to_workspace(inst2pts, workspace_voxels)
    n_ws = len(out)
    out = filter_by_frame_count(out, node2inst, min_frames=min_frames)
    n_mf = len(out)

    table_id = -1
    if remove_table:
        table_id = determine_table_instance(out, scene.ground_plane)
        if table_id != -1:
            out.pop(table_id, None)
            logger.info("table: removed instance %d", table_id)

    logger.info(
        "deg filters: %d in -> %d after workspace -> %d after min-frame -> %d after table",
        n0, n_ws, n_mf, len(out),
    )
    return out, table_id

"""Geometric substrate reconstruction for the instance-tree builder.

The public Clutt3R-Seg release ships ``initial_segmenter.py`` (the *consumer* of a
precomputed ``instance_tree.json``) but not the builder that produces that
artifact. To rebuild the artifact for new sequences we have to recover the same
geometric substrate the consumer uses, namely:

* a back-projected, voxel-downsampled world point cloud,
* a normal/distance super-voxel oversegmentation (union-find on a kNN graph), and
* per-mask super-voxel occupancy statistics.

This module reproduces exactly that part of
``initial_segmenter.initial_segmentation_consistency`` (same 5 mm voxels, same
``d = 0.5*dist + (1 - n_i . n_j)`` super-voxel metric, same occupancy counting) so
the spatial affinities used by :mod:`clutt3rseg.tree_builder` are computed on the
same kind of super-voxel substrate the consumer later votes on. The downstream
consumer is left untouched; it re-derives its own substrate from the masks and
the ``leaf2inst`` mapping this builder writes.

Both this builder and the consumer (``initial_segmenter``) tolerate depth holes:
back-projection keeps only valid pixels and the per-mask code maps the rest to
-1 and filters them with a consistent global offset, so measured/sensor depth
(with invalid pixels) works without the dense MVSAnywhere depth the paper assumes
-- keeping the baseline comparable to the others, which use measured depth.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import glob
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image

from .utils import _bbox, _letterbox, backproject, load_depth, refine_masks_by_depth

logger = logging.getLogger("clutt3rseg-builder")

FRAME_OFFSET = 100

# Super-voxel grouping constants, matching initial_segmenter.initial_segmentation_consistency.
SUPERVOXEL_KNN = 8
ALPHA_DIST = 0.5  # weight on Euclidean distance in the super-voxel merge metric
BETA_NORMAL = 1.0  # weight on (1 - normal agreement) in the super-voxel merge metric

# Leaf = (frame index, Grounded-SAM mask id). Mask ids are 1-indexed as in the files.
Leaf = tuple[int, int]


@dataclass
class SceneSubstrate:
    """Everything the tree builder needs that depends on geometry/appearance."""

    frame_masks: dict[int, dict[int, np.ndarray]]
    """frame index -> {mask id -> boolean HxW mask} (the leaves available per frame)."""

    crops: dict[Leaf, np.ndarray]
    """leaf -> 224x224 letterboxed RGB crop for DuoduoCLIP (only for masks with >10 px)."""

    mask2sp_counts: dict[Leaf, dict[int, int]]
    """leaf -> {super-voxel id -> number of that super-voxel's down-voxels the mask covers}."""

    counts_per_sp: np.ndarray
    """super-voxel id -> number of down-voxels it contains."""

    ground_sp_id: int
    """id of the largest super-voxel (treated as ground/table, as the consumer does)."""

    n_supervoxels: int

    meta: dict = field(default_factory=dict)
    """transforms.json contents (intrinsics + frames), for downstream reuse."""


def _intrinsics(meta: dict) -> np.ndarray:
    return np.array(
        [[meta["fl_x"], 0, meta["cx"]], [0, meta["fl_y"], meta["cy"]], [0, 0, 1]],
        np.float32,
    )


def _supervoxel_labels(xyz: np.ndarray, normals: np.ndarray, gc_lambda: float) -> np.ndarray:
    """Union-find super-voxel oversegmentation on a kNN graph.

    Mirrors the consumer: contract neighbours whose ``0.5*dist + (1 - dot)`` falls
    below ``gc_lambda`` (paper: alpha=0.5, beta=1.0, tau_merge=0.01).
    """
    n = len(xyz)
    parent = np.arange(n)
    size = np.ones(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    tree = o3d.geometry.KDTreeFlann(pcd)

    for i, pt in enumerate(xyz):
        _, idx, _ = tree.search_knn_vector_3d(pt, SUPERVOXEL_KNN + 1)
        nbr = np.asarray(idx, dtype=np.int32)[1:]
        if nbr.size == 0:
            continue
        di = find(i)
        ni = normals[i]
        for j in nbr:
            dj = find(j)
            if di == dj:
                continue
            v = xyz[j] - pt
            dist = float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
            nj = normals[j]
            dot = float(ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2])
            w = ALPHA_DIST * dist + BETA_NORMAL * (1.0 - dot)
            if w < gc_lambda:
                if size[di] < size[dj]:
                    di, dj = dj, di
                parent[dj] = di
                size[di] += size[dj]
                di = find(di)

    root2lbl: dict[int, int] = {}
    labels = np.empty(n, np.int32)
    cur = 0
    for i in range(n):
        r = find(i)
        if r not in root2lbl:
            root2lbl[r] = cur
            cur += 1
        labels[i] = root2lbl[r]
    return labels


def build_scene_substrate(
    experiment_data_dir: Path,
    initial_idx: list[int],
    *,
    voxel_size: float = 0.005,
    depth_scale: float = 0.001,
    max_depth: float = 5.0,
    gc_lambda: float = 0.010,
    min_points: int = 3,
) -> SceneSubstrate:
    data_dir = Path(experiment_data_dir) / "data"
    meta = json.load(open(data_dir / "transforms.json"))
    frames_meta = meta["frames"]
    K = _intrinsics(meta)
    instance_mask_path = data_dir / "instance_masks"

    frame_masks: dict[int, dict[int, np.ndarray]] = {}
    crops: dict[Leaf, np.ndarray] = {}
    inst2pids: dict[Leaf, list[int]] = {}

    xyz_full_parts, rgb_full_parts = [], []
    offset = 0

    for f_idx in initial_idx:
        fr = frames_meta[f_idx]
        rgb_path = data_dir / Path(fr["file_path"]).with_suffix(".png")
        depth_path = data_dir / Path(fr["depth_file_path"]).with_suffix(".png")
        if not rgb_path.exists():
            logger.warning("Frame %d: missing RGB %s, skipping", f_idx, rgb_path)
            continue
        if not depth_path.exists():
            logger.warning("Frame %d: missing depth %s, skipping", f_idx, depth_path)
            continue

        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB) / 255.0
        raw_image = Image.open(rgb_path).convert("RGB")
        H, W = rgb.shape[:2]
        depth = load_depth(depth_path, depth_scale)
        pts_cam, uvs = backproject(depth, K, max_depth)
        T = np.asarray(fr["transform_matrix"], np.float32)
        pts_wld = (T[:3, :3] @ pts_cam.T + T[:3, 3:4]).T

        h, w = depth.shape
        idx_img = np.full((h, w), -1, np.int32)
        idx_img[uvs[:, 1], uvs[:, 0]] = np.arange(len(uvs))

        mask_glob = glob.glob(str(instance_mask_path / f"mask_{f_idx:06d}_*.png"))
        masks = refine_masks_by_depth(mask_glob, depth, mode="naive")
        frame_masks[f_idx] = masks

        for lid, mask in masks.items():
            if mask.sum() > 10:
                y0, y1, x0, x1 = _bbox(mask)
                crop_rgb = raw_image.crop((x0, y0, x1 + 1, y1 + 1))
                crop_mask = (mask[y0 : y1 + 1, x0 : x1 + 1] * 255).astype(np.uint8)
                crops[(f_idx, lid)] = _letterbox(crop_rgb, crop_mask)

            ys, xs = np.where(mask)
            if ys.size:
                idx_pts = idx_img[ys, xs]
                valid = idx_pts != -1
                if np.any(valid):
                    inst2pids[(f_idx, lid)] = (offset + idx_pts[valid]).tolist()

        xyz_full_parts.append(pts_wld)
        rgb_full_parts.append(rgb[uvs[:, 1], uvs[:, 0]])
        offset += len(pts_wld)
        if len(pts_wld) != H * W:
            invalid = H * W - len(pts_wld)
            logger.warning(
                "Frame %d: %d invalid/holed depth pixels (%.1f%%); proceeding on valid pixels only (measured depth).",
                f_idx, invalid, 100.0 * invalid / (H * W),
            )

    if not xyz_full_parts:
        raise RuntimeError("No initial frames produced points; check RGB/depth paths and initial_idx.")

    xyz_full = np.concatenate(xyz_full_parts, 0)
    rgb_full = np.concatenate(rgb_full_parts, 0)

    # ---- custom voxel downsample (same aggregation/min-occupancy as the consumer) ----
    vox = np.floor(xyz_full / voxel_size).astype(np.int32)
    df = pd.DataFrame(
        {
            "vx": vox[:, 0], "vy": vox[:, 1], "vz": vox[:, 2],
            "px": xyz_full[:, 0], "py": xyz_full[:, 1], "pz": xyz_full[:, 2],
            "cr": rgb_full[:, 0], "cg": rgb_full[:, 1], "cb": rgb_full[:, 2],
        }
    )
    agg = df.groupby(["vx", "vy", "vz"]).agg(
        {"px": "sum", "py": "sum", "pz": "sum", "cr": "sum", "cg": "sum", "cb": "sum", "vx": "size"}
    ).rename(columns={"vx": "counts"})
    filtered = agg[agg["counts"] >= min_points].copy()
    n_down = len(filtered)
    counts = filtered["counts"].values
    down_pts = (filtered[["px", "py", "pz"]].values / counts[:, None]).astype(np.float32)

    filtered["new_idx"] = np.arange(n_down)
    df = df.join(filtered["new_idx"], on=["vx", "vy", "vz"])
    orig2down = df["new_idx"].fillna(-1).to_numpy(dtype=np.int32)

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(down_pts)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel_size, max_nn=30))
    normals_down = np.asarray(pcd_down.normals)

    # ---- super-voxels ----
    down2sp = _supervoxel_labels(down_pts.astype(np.float64), normals_down, gc_lambda)
    n_sp = int(down2sp.max()) + 1
    _, counts_per_sp = np.unique(down2sp, return_counts=True)
    ground_sp_id = int(np.argmax(counts_per_sp))

    # ---- per-mask super-voxel occupancy (unique down-voxels per super-voxel) ----
    mask2sp_counts: dict[Leaf, dict[int, int]] = defaultdict(dict)
    for leaf, pids in inst2pids.items():
        if not pids:
            continue
        down = orig2down[np.asarray(pids)]
        down = down[down != -1]
        if down.size == 0:
            continue
        sp_ids = down2sp[np.unique(down)]
        uq, cnts = np.unique(sp_ids, return_counts=True)
        mask2sp_counts[leaf] = dict(zip(uq.tolist(), cnts.tolist()))

    return SceneSubstrate(
        frame_masks=frame_masks,
        crops=crops,
        mask2sp_counts=dict(mask2sp_counts),
        counts_per_sp=counts_per_sp,
        ground_sp_id=ground_sp_id,
        n_supervoxels=n_sp,
        meta=meta,
    )

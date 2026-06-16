"""Shared geometry, mask, and point-cloud utilities."""
import json, re, glob
from pathlib import Path
from functools import lru_cache
import colorsys
import numpy as np
import cv2
import open3d as o3d
from typing import Dict, Tuple
from collections import defaultdict

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        return 1 - ((2. * intersection + self.smooth) /
                    (input_flat.sum() + target_flat.sum() + self.smooth))

try:
    import matplotlib.colors
except ImportError:
    raise ImportError("matplotlib is required for distinct color palette.")


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("Empty mask")
    return ys.min(), ys.max(), xs.min(), xs.max()

def interpolate_to_patch_size(img_bchw, patch_size):

    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W

def distinct_palette(n: int) -> np.ndarray:
    hsv = np.stack([np.linspace(0, 1, n, endpoint=False),
                    np.full(n, 0.75),
                    np.full(n, 0.95)], axis=1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb

from sklearn.decomposition import PCA
def pca_rgb(feat_bhwc):
    H,W,C = feat_bhwc.shape
    f = feat_bhwc.reshape(-1, C)
    rgb = PCA(n_components=3).fit_transform(f)
    rgb = (rgb - rgb.min(0)) / (rgb.max(0)-rgb.min(0) + 1e-5)
    return (rgb.reshape(H, W, 3)*255).astype('uint8')

def _letterbox(rgb: Image.Image, bin_mask: np.ndarray) -> np.ndarray:
    tw, th = (224, 224)
    ow, oh = rgb.size
    scale = min(tw / ow, th / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))

    rgb_rs  = rgb.resize((nw, nh), Image.BILINEAR)
    mask_rs = Image.fromarray(bin_mask).resize((nw, nh), Image.NEAREST)

    canvas = np.full((th, tw, 3), 255, np.uint8)
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas[top : top + nh, left : left + nw] = np.array(rgb_rs)

    mask_bool = np.zeros((th, tw), bool)
    mask_bool[top : top + nh, left : left + nw] = np.array(mask_rs) > 0
    canvas[~mask_bool] = 255
    return canvas

def load_depth(path: Path, scale: float):
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return img * scale


def prompt_to_ply_name(prompt: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt.strip()).strip("._-")
    if not name:
        name = "target"
    return f"{name}.ply"


def export_prompt_pointcloud(points: np.ndarray, colors: np.ndarray, output_dir: Path, prompt: str) -> Path:
    if points is None or len(points) == 0:
        raise RuntimeError(f"No points to export for prompt '{prompt}'.")

    colors = colors.astype(np.float64)
    if colors.size and colors.max() > 1.0:
        colors = colors / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / prompt_to_ply_name(prompt)
    o3d.io.write_point_cloud(str(export_path), pcd)
    return export_path


def K_from_meta(meta: dict, frame: dict | None = None) -> np.ndarray:
    """Camera intrinsics matrix, preferring per-frame ``K`` over a shared top-level one.

    The bundled samples store one set of top-level ``fl_x/fl_y/cx/cy`` shared by all
    frames; deg datasets (and any multi-camera rig) instead carry a per-frame 3x3
    ``K``. Supporting both lets the same pipeline run on either.
    """
    if frame is not None and "K" in frame:
        return np.asarray(frame["K"], np.float32).reshape(3, 3)
    if "fl_x" in meta:
        return np.array([[meta["fl_x"], 0, meta["cx"]], [0, meta["fl_y"], meta["cy"]], [0, 0, 1]], np.float32)
    raise KeyError("transforms.json has neither a per-frame 'K' nor top-level 'fl_x' intrinsics.")


def backproject(depth, K, max_depth):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    valid = (depth > 0) & (depth < max_depth)
    z = depth[valid]
    x = (xs[valid] - K[0,2]) * z / K[0,0]
    y = (ys[valid] - K[1,2]) * z / K[1,1]
    return np.stack([x, y, z], 1), np.stack([xs[valid], ys[valid]], 1)

@lru_cache(maxsize=None)
def vivid_color(idx: int):
    if idx <= 0:
        return np.array([.8, .8, .8], np.float32)
    h = (idx * 0.618033988749895) % 1.0
    return np.array(colorsys.hsv_to_rgb(h, .85, .90), np.float32)


def refine_masks_by_depth(
    mask_glob: list[str],
    depth: np.ndarray,
    mode: str = "refine",
) -> Dict[int, np.ndarray]:
    mask_refined = {}

    if mode == "naive":
        for m in mask_glob:
            inst = int(re.findall(r"mask_\d{6}_(\d+)\.png", m)[0])
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE) > 0
            mask_refined[inst] = mask
        return mask_refined

    elif mode == "refine":


        raise NotImplementedError("Mode 'refine' is not implemented yet.")


    raise NotImplementedError(f"Mode '{mode}' not implemented for mask refinement.")


def load_update_gt_data(dataset_root, fidx, verbose: bool = True):
    try:

        meta = json.load(open(dataset_root / "transforms.json"))
        frames_meta = meta["frames"]


        frame_meta = None
        for i, fr in enumerate(frames_meta):
            if i == fidx:
                frame_meta = fr
                break

        if frame_meta is None:
            return None


        rgb_path = dataset_root / Path(frame_meta["file_path"]).with_suffix(".png")
        depth_path = dataset_root / Path(frame_meta["depth_file_path"]).with_suffix(".png")
        if not rgb_path.exists():
            return None

        img = Image.open(rgb_path).convert("RGB")
        img_tensor = T.ToTensor()(img)


        K = np.array([[meta["fl_x"], 0, meta["cx"]],
                      [0, meta["fl_y"], meta["cy"]],
                      [0, 0, 1]], np.float32)


        transform_matrix = np.array(frame_meta["transform_matrix"])
        pose = np.linalg.inv(transform_matrix)


        img_shape = img.size[::-1]

        depth = load_depth(depth_path, 0.001)

        return {
            'fidx': fidx,
            'K': K,
            'pose': pose,
            'image_tensor': img_tensor,
            'img_shape': img_shape,
            'depth': depth
        }

    except Exception:
        return None


def _load_w2c_and_size(frames_meta, fidx: int):
    c2w = np.asarray(frames_meta[fidx]["transform_matrix"], np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)

    return w2c

def _project_points(xyz, K, w2c, H, W):
    pts_cam = (w2c[:3, :3] @ xyz.T + w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    valid = z > 1e-4
    u = pts_cam[:, 0] * K[0, 0] / z + K[0, 2]
    v = pts_cam[:, 1] * K[1, 1] / z + K[1, 2]
    valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, valid

def mask_filter_dense_instances(
    export_inst2all_points: dict[int, np.ndarray],
    leaf2inst_graph: dict[tuple[int, int], int],
    frames_meta: list,
    instance_mask_path: Path,
    K: np.ndarray,
    *,
    erode_px: int = 2,
    min_view_count: int = 1,
    min_view_ratio: float = 0.5,
    verbose: bool = True,
) -> dict[int, np.ndarray]:

    inst2leaves = defaultdict(list)
    for (f, l), inst in leaf2inst_graph.items():
        inst2leaves[inst].append((f, l))


    frame_cache = {}
    all_fidx = {f for leaves in inst2leaves.values() for f, l in leaves}
    kernel = np.ones((erode_px, erode_px), np.uint8) if erode_px > 0 else None

    for fidx in all_fidx:

        masks = {}
        mask_files = glob.glob(str(instance_mask_path / f"mask_{fidx:06d}_*.png"))
        for mpath in mask_files:
            lid = int(Path(mpath).stem.split('_')[-1])
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = (m > 0).astype(np.uint8)
                if kernel is not None:
                    m = cv2.erode(m, kernel, iterations=1)
                masks[lid] = m

        fmeta = frames_meta[fidx]
        K_f = np.asarray(fmeta["K"], np.float32).reshape(3, 3) if "K" in fmeta else K
        frame_cache[fidx] = {
            'w2c': np.linalg.inv(np.array(fmeta['transform_matrix'])),
            'K': K_f,
            'masks': masks,
            'size': masks[list(masks.keys())[0]].shape if masks else (0,0)
        }

    filtered_instances = {}
    for inst_id, pts in export_inst2all_points.items():
        if pts.size == 0: continue
        xyz = pts[:, :3]


        votes = np.zeros(xyz.shape[0], dtype=np.int32)
        visible_counts = np.zeros_like(votes)

        leaves = inst2leaves.get(inst_id, [])
        if not leaves:
            filtered_instances[inst_id] = pts
            continue


        supporting_fidx = {f for f, l in leaves}
        for fidx in supporting_fidx:
            cache = frame_cache.get(fidx)
            if not cache or not cache['masks']: continue

            w2c, H, W = cache['w2c'], cache['size'][0], cache['size'][1]
            u, v, z, is_visible = _project_points(xyz, cache['K'], w2c, H, W)

            visible_counts += is_visible.astype(np.int32)

            if not np.any(is_visible): continue


            union_mask = np.zeros((H, W), dtype=np.uint8)
            for f, l in leaves:
                if f == fidx and l in cache['masks']:
                    union_mask |= cache['masks'][l]


            ui = u[is_visible].astype(np.int32)
            vi = v[is_visible].astype(np.int32)


            ui = np.clip(ui, 0, W - 1)
            vi = np.clip(vi, 0, H - 1)

            hits = (union_mask[vi, ui] > 0)
            votes[is_visible] += hits.astype(np.int32)


        required_votes = np.maximum(min_view_count, np.ceil(min_view_ratio * np.maximum(visible_counts, 1))).astype(np.int32)
        keep_mask = votes >= required_votes

        if np.any(keep_mask):
            filtered_instances[inst_id] = pts[keep_mask]

    return filtered_instances

def filter_invalid_instances_by_superpoint_stats(
    sp2inst: dict[int, int],
    counts_per_sp: np.ndarray,
    *,
    min_down_points: int = 80,
    min_unique_sp: int = 3,
    min_orig_points: int | None = None,
    export_after_mask: dict[int, np.ndarray] | None = None,
    verbose: bool = True,
) -> set[int]:
    if not sp2inst:
        return set()

    items = [(sp, inst) for sp, inst in sp2inst.items() if inst >= 0]
    if not items:
        return set()


    df = pd.DataFrame(items, columns=['sp_id', 'inst_id'])


    df['sp_counts'] = counts_per_sp[df['sp_id'].values]


    stats = df.groupby('inst_id').agg(
        n_unique_sp=('sp_id', 'nunique'),
        total_down_points=('sp_counts', 'sum')
    )


    mask_sp_stats = (stats['n_unique_sp'] < min_unique_sp) |\
                    (stats['total_down_points'] < min_down_points)

    drop = set(stats[mask_sp_stats].index)

    if (min_orig_points is not None) and (export_after_mask is not None):

        candidate_ids = stats.index.difference(drop)
        for inst_id in candidate_ids:
            n_orig = export_after_mask.get(inst_id, np.empty((0,6))).shape[0]
            if n_orig < min_orig_points:
                drop.add(inst_id)
    return drop

def reconstruct_instances_from_superpoints(
    xyz_full: np.ndarray,
    rgb_full: np.ndarray,
    orig2down: np.ndarray,
    down2sp: np.ndarray,
    sp2inst: dict[int, int],
) -> tuple[dict[int, np.ndarray], np.ndarray]:


    sp_map_size = down2sp.max() + 1 if down2sp.size > 0 else 0
    sp_to_final_inst = np.full(sp_map_size, -1, dtype=np.int32)
    if sp2inst:
        valid_sp_indices = np.array(list(sp2inst.keys()))
        valid_inst_values = np.array(list(sp2inst.values()))
        sp_to_final_inst[valid_sp_indices] = valid_inst_values


    orig_inst_labels = np.full(len(xyz_full), -1, dtype=np.int32)
    valid_orig_mask = (orig2down != -1)
    valid_indices = orig2down[valid_orig_mask]
    orig_inst_labels[valid_orig_mask] = sp_to_final_inst[down2sp[valid_indices]]


    export_inst2all_points = {}
    unique_inst_ids = np.unique(orig_inst_labels)
    for inst_id in unique_inst_ids:
        if inst_id == -1: continue
        inst_mask = (orig_inst_labels == inst_id)
        points = xyz_full[inst_mask]
        colors = rgb_full[inst_mask]
        export_inst2all_points[inst_id] = np.hstack([points, colors])


    return export_inst2all_points, orig_inst_labels

def load_update_frame_data(dataset_root, fidx, verbose: bool = True):
    try:

        meta = json.load(open(dataset_root / "transforms.json"))
        frames_meta = meta["frames"]


        frame_meta = None
        for i, fr in enumerate(frames_meta):
            if i == fidx:
                frame_meta = fr
                break

        if frame_meta is None:
            return None


        rgb_path = dataset_root / Path(frame_meta["file_path"]).with_suffix(".png")
        if not rgb_path.exists():
            return None

        img = Image.open(rgb_path).convert("RGB")
        img_tensor = T.ToTensor()(img)


        K = np.array([[meta["fl_x"], 0, meta["cx"]],
                      [0, meta["fl_y"], meta["cy"]],
                      [0, 0, 1]], np.float32)


        transform_matrix = np.array(frame_meta["transform_matrix"])
        pose = np.linalg.inv(transform_matrix)


        img_shape = img.size[::-1]

        return {
            'fidx': fidx,
            'K': K,
            'pose': pose,
            'image_tensor': img_tensor,
            'img_shape': img_shape
        }

    except Exception:
        return None

def so3_exp(r):
    theta = torch.norm(r) + 1e-9
    k = r/theta
    K = torch.zeros((3, 3), device=r.device, dtype=r.dtype)
    K = torch.tensor([[0,-k[2],k[1]],
                      [k[2],0,-k[0]],
                      [-k[1],k[0],0]], device=r.device)
    R = torch.eye(3, device=r.device, dtype=r.dtype) +\
        torch.sin(theta)*K + (1-torch.cos(theta))*(K@K)
    return R

from scipy import ndimage
import torch.nn.functional as F


def r6d_to_mat(x):
    a1 = F.normalize(x[..., 0:3], dim=-1)
    a2 = F.normalize(x[..., 3:6] - (a1*(x[..., 3:6]*a1).sum(-1, keepdim=True)), dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack([a1, a2, a3], dim=-2)

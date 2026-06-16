"""Adapter from a deg dataset (transforms.json + StaticDataset) to the Clutt3R-Seg
workspace layout.

The other deg baselines consume a deg ``transforms.json`` via
``load_observations_from_transforms_path`` (RGB + depth + per-frame pose/K).
Clutt3R-Seg instead expects a ``data/`` workspace
(``transforms.json`` + ``images/`` + ``depth/`` + ``instance_masks/``). This
module materialises that workspace from the deg observations so the same
segmenter can run on the eval datasets, handling the two real differences:

* **Coordinate convention.** deg poses (``X_WV``) follow the OpenGL/NeRF
  convention (the deg ``Frame`` derives ``X_VW_opencv`` by negating the Y/Z rows).
  Clutt3R-Seg back-projects with an OpenCV pinhole, so each pose is converted to
  OpenCV cam-to-world: ``T_cv = X_WV @ diag(1, -1, -1, 1)`` (negate rotation
  columns 1 and 2). Verified against ``psdframe.Frame.X_VW_opencv``.

* **Per-frame intrinsics.** deg datasets are often multi-camera rigs with a
  different ``K`` per frame (not always 6 cameras). Each frame's ``K`` is written
  through; the consumer/substrate resolve it per frame (see
  ``utils.K_from_meta``).

Depth is written as 16-bit millimetres (``depth_scale = 1e-3``); invalid/holed
pixels become 0 and are tolerated downstream (the dense-depth requirement was
loosened). Masks (``instance_masks/``) and the instance tree are produced later
by the Grounded-SAM step and the tree builder.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("clutt3rseg-adapter")

# OpenGL/NeRF cam -> OpenCV cam (negate Y and Z axes).
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

DEPTH_SCALE = 1e-3  # we store depth as uint16 millimetres
_MAX_DEPTH_MM = 65535


def is_deg_dataset(transforms_path: Path) -> bool:
    """True for a deg transforms.json (per-frame K / depth_path), False for the
    Clutt3R-Seg sample layout (top-level fl_x / depth_file_path under data/)."""
    try:
        meta = json.loads(Path(transforms_path).read_text())
    except Exception:
        return False
    frames = meta.get("frames", [])
    if not frames:
        return False
    f0 = frames[0]
    if "depth_file_path" in f0:  # Clutt3R-Seg native layout
        return False
    return ("K" in f0) or ("depth_path" in f0)


def materialize_workspace(transforms_path: Path, workspace_dir: Path, *, overwrite_images: bool = False) -> int:
    """Write a Clutt3R-Seg ``data/`` workspace from a deg dataset. Returns #frames."""
    from initializerdefs import load_observations_from_transforms_path

    obs = load_observations_from_transforms_path(transforms_path)
    data_dir = Path(workspace_dir) / "data"
    images_dir = data_dir / "images"
    depth_dir = data_dir / "depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    frames_out = []
    for i, fr in enumerate(obs.frames):
        if fr.depth is None:
            raise ValueError(
                f"Frame {i} ({fr.name}) has no depth. Clutt3R-Seg is an RGB-D method; "
                "the dataset must provide depth."
            )

        rgb_name = f"image_{i:06d}.png"
        depth_name = f"depth_{i:06d}.png"
        rgb_path = images_dir / rgb_name
        depth_path = depth_dir / depth_name

        if overwrite_images or not rgb_path.exists():
            rgb_u8 = np.clip(np.asarray(fr.color) * 255.0, 0, 255).astype(np.uint8)  # [h,w,3] RGB
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

        if overwrite_images or not depth_path.exists():
            depth_m = np.asarray(fr.depth, dtype=np.float32)
            depth_mm = depth_m * 1000.0
            valid = np.isfinite(depth_mm) & (depth_mm > 0) & (depth_mm <= _MAX_DEPTH_MM)
            depth_u16 = np.where(valid, depth_mm, 0).astype(np.uint16)
            cv2.imwrite(str(depth_path), depth_u16)

        T_cv = (np.asarray(fr.X_WV, dtype=np.float32) @ _GL_TO_CV)
        h, w = np.asarray(fr.color).shape[:2]
        frames_out.append({
            "name": fr.name,
            "file_path": f"images/{rgb_name}",
            "depth_file_path": f"depth/{depth_name}",
            "transform_matrix": T_cv.tolist(),
            "K": np.asarray(fr.K, dtype=np.float32).reshape(3, 3).tolist(),
            "w": int(w),
            "h": int(h),
        })

    (data_dir / "transforms.json").write_text(json.dumps({"depth_scale": DEPTH_SCALE, "frames": frames_out}, indent=2))
    logger.info("Materialized %d frames into %s", len(frames_out), data_dir)
    return len(frames_out)

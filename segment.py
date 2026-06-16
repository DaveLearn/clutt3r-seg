"""deg external-segmenter entry point for Clutt3R-Seg.

Invoked by the deg pipeline (see ``ExternalSegmentationInitializer``) as::

    python segment.py <transforms_path> <scene_path> [--key value ...]

Clutt3R-Seg is a language-grounded pipeline whose public release ships *without*
its instance-tree builder, and it additionally requires per-frame Grounded-SAM
``instance_masks/`` and a precomputed ``instance_tree.json`` next to the
sequence's ``transforms.json``. This glue therefore expects a Clutt3R-Seg-style
sequence layout::

    <experiment_data_dir>/data/transforms.json
    <experiment_data_dir>/data/images/...
    <experiment_data_dir>/data/depth/...            (dense MVSAnywhere depth)
    <experiment_data_dir>/data/instance_masks/mask_<frame>_<inst>.png
    <experiment_data_dir>/data/instance_tree.json

It runs the upstream initial-segmentation pipeline and exports *all* recovered
instances (not just the prompt-matched target) as a class-agnostic
``ObjectSegmentations``, so the result is comparable to the other baselines.

The ``scene_path`` positional is accepted for contract compatibility but unused:
Clutt3R-Seg does not consume the deg ``SceneSetup``.

Outputs ``objects_path: <path>`` on stdout for the parent process to read.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import tyro

from initializerdefs import ObjectSegmentations, PointCloudObjectDef

from clutt3rseg.clip_backends.duoduo import DEFAULT_DUODUO_CHECKPOINT, load_duoduo_clip
from clutt3rseg.initial_segmenter import (
    initial_segmentation_consistency,
    validate_initial_dataset,
)
from clutt3rseg.tree_artifacts import ARTIFACT_NAME, load_instance_tree_artifact
from clutt3rseg.tree_builder import build_instance_tree_artifact, write_instance_tree_artifact


def _patch_open3d_pathlike() -> None:
    """open3d 0.18 (pinned here by the numpy<2 requirement of initializerdefs) only
    accepts str filenames, but upstream initial_segmenter passes pathlib.Path. Coerce
    the first arg to str so the vendored fork can stay unmodified."""
    _orig_write = o3d.io.write_point_cloud

    def _write_point_cloud(filename, *args, **kwargs):
        return _orig_write(str(filename), *args, **kwargs)

    o3d.io.write_point_cloud = _write_point_cloud


_patch_open3d_pathlike()


def _parse_initial_idx(value: str) -> list[int]:
    """Parse "0,1,2" / "0 1 2" into a list of ints."""
    tokens = [tok for tok in value.replace(",", " ").split() if tok]
    return [int(tok) for tok in tokens]


def _experiment_dir_from_transforms(transforms_path: Path) -> Path:
    """Clutt3R-Seg expects ``<experiment_data_dir>/data/transforms.json``."""
    data_dir = transforms_path.parent
    if data_dir.name != "data":
        raise ValueError(
            f"Expected transforms.json inside a 'data/' directory (Clutt3R-Seg layout), got {transforms_path}. "
            "Clutt3R-Seg needs data/transforms.json alongside data/instance_masks/ and data/instance_tree.json."
        )
    return data_dir.parent


@dataclass
class Args:
    transforms_path: tyro.conf.Positional[Path]
    """Path to <experiment_data_dir>/data/transforms.json."""

    scene_path: tyro.conf.Positional[Path]
    """Path to the pickled deg SceneSetup (accepted for contract compatibility; unused)."""

    initial_idx: Optional[str] = None
    """Comma/space separated initial frame indices. Defaults to instance_tree.json's initial_idx.
    Required when no instance_tree.json exists and it has to be built."""

    build_tree_if_missing: bool = True
    """If data/instance_tree.json is absent, reconstruct it with the paper's tree builder
    (clutt3rseg.tree_builder) instead of failing. Requires --initial-idx."""

    update_idx: Optional[str] = None
    """When building the tree, also emit containment trees for these update frames."""

    target_prompt: str = "object"
    """Language prompt for the upstream target export. The deg glue exports all instances regardless."""

    voxel_size: float = 0.005
    """Voxel size for superpoint downsampling."""

    depth_scale: float = 0.001
    """Multiplier converting stored depth units to metres."""

    max_depth: float = 5.0
    """Maximum valid depth in metres."""

    gc_lambda: float = 0.010
    """Superpoint graph-construction threshold (lower = more edges)."""

    min_points: int = 3
    """Minimum original points per voxel to keep during downsampling."""

    clip_bs: int = 1
    """DuoduoCLIP image-encoder batch size."""

    estimate_normals: bool = True
    """Estimate per-instance normals (PointCloudObjectDef requires a normals array)."""

    duoduo_root: Optional[Path] = None
    """External DuoduoCLIP checkout. Defaults to $DUODUOCLIP_ROOT."""

    clip_checkpoint: str = DEFAULT_DUODUO_CHECKPOINT
    """DuoduoCLIP checkpoint filename/path."""


def _to_point_cloud_objects(inst2all_points: dict[int, np.ndarray], estimate_normals: bool) -> list[PointCloudObjectDef]:
    objects: list[PointCloudObjectDef] = []
    for out_id, inst_id in enumerate(sorted(inst2all_points.keys())):
        arr = np.asarray(inst2all_points[inst_id])
        if arr.size == 0:
            continue
        points = arr[:, :3].astype(np.float32)
        colors = arr[:, 3:6].astype(np.float32)

        if estimate_normals and len(points) >= 3:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=16))
            normals = np.asarray(pcd.normals, dtype=np.float32)
        else:
            normals = np.zeros_like(points)

        objects.append(
            PointCloudObjectDef(
                object_id=int(out_id),
                points=points,
                normals=normals,
                color=colors,
            )
        )
    return objects


def run() -> None:
    logger = logging.getLogger("clutt3rseg-segmenter")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)-18s: %(levelname)-8s %(message)s"))
    logger.addHandler(handler)

    args = tyro.cli(Args)

    transforms_path = args.transforms_path.resolve()
    experiment_data_dir = _experiment_dir_from_transforms(transforms_path)

    output_dir = experiment_data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_path = experiment_data_dir / "data" / ARTIFACT_NAME

    # All upstream chatter goes to stderr; only the final objects_path line is on stdout.
    with contextlib.redirect_stdout(sys.stderr):
        logger.info("Starting Clutt3R-Seg initialization")
        logger.info("transforms_path=%s", transforms_path)
        logger.info("experiment_data_dir=%s", experiment_data_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading DuoduoCLIP (device=%s) ...", device)
        clip = load_duoduo_clip(
            checkpoint=args.clip_checkpoint,
            device=device,
            duoduo_root=args.duoduo_root,
        )

        # The public release ships no instance-tree builder; reconstruct it from the
        # paper (clutt3rseg.tree_builder) when the artifact is missing so the pipeline
        # can run on new sequences, not just the bundled samples.
        if not tree_path.exists() and args.build_tree_if_missing:
            if args.initial_idx is None:
                raise ValueError(
                    f"{tree_path} is missing and must be built, but no --initial-idx was given. "
                    "Pass e.g. --initial-idx '0,1,2,3,4,5,6,7'."
                )
            build_idx = _parse_initial_idx(args.initial_idx)
            update_idx = _parse_initial_idx(args.update_idx) if args.update_idx else None
            logger.info("No instance_tree.json found; building it for initial_idx=%s ...", build_idx)
            artifact = build_instance_tree_artifact(
                experiment_data_dir,
                build_idx,
                clip=clip,
                update_idx=update_idx,
                voxel_size=args.voxel_size,
                depth_scale=args.depth_scale,
                max_depth=args.max_depth,
                gc_lambda=args.gc_lambda,
                min_points=args.min_points,
            )
            write_instance_tree_artifact(experiment_data_dir, artifact)
            logger.info("Wrote %s", tree_path)

        if args.initial_idx is not None:
            initial_idx = _parse_initial_idx(args.initial_idx)
        else:
            artifact = load_instance_tree_artifact(experiment_data_dir)
            initial_idx = [int(i) for i in artifact.get("initial", {}).get("initial_idx", [])]
            if not initial_idx:
                raise ValueError(
                    "No --initial-idx given and instance_tree.json has no initial.initial_idx. "
                    "Pass --initial-idx '0,1,2,3,4,5,6,7'."
                )
        logger.info("initial_idx=%s", initial_idx)

        run_args = SimpleNamespace(
            experiment_data_dir=experiment_data_dir,
            initial_idx=initial_idx,
            target_prompt=args.target_prompt,
            voxel_size=args.voxel_size,
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            gc_lambda=args.gc_lambda,
            min_points=args.min_points,
            clip_bs=args.clip_bs,
        )

        validate_initial_dataset(run_args)

        logger.info("Running initial segmentation ...")
        initial_data = initial_segmentation_consistency(run_args, clip=clip, device=device)

        inst2all_points = initial_data["original_data"]["inst2all_points"]
        logger.info("Recovered %d instances", len(inst2all_points))

        objects = ObjectSegmentations(
            object_segmentations=_to_point_cloud_objects(inst2all_points, args.estimate_normals)
        )

        out_root = Path(__file__).parent / "outputs"
        out_dir = out_root / f"{time.strftime('%Y%m%d-%H%M%S')}_{experiment_data_dir.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        objects_path = out_dir / "objectsdef.pkl"
        objects.save(objects_path)
        logger.info("Saved %d objects to %s", len(objects.object_segmentations), objects_path)

    print(f"objects_path: {objects_path}")


if __name__ == "__main__":
    run()

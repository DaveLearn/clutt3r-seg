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
``ObjectSegmentations``, so the result is comparable to the other baselines. For
deg datasets it emits per-frame ``InstanceMaskObjectsDef`` (the form the deg
harness requires, matching SAM3D / MaskClustering / Open3DIS / SAI3D); for the
native clutt3r samples it emits the point-cloud instances.

The ``scene_path`` positional is the deg ``SceneSetup`` pickle, consumed by the
parity filters (workspace crop / min-frame / table removal) on deg datasets; the
native clutt3r samples have no ``SceneSetup`` and ignore it.

Outputs ``objects_path: <path>`` on stdout for the parent process to read.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import open3d as o3d
import torch
import tyro

from initializerdefs import (
    InstanceMaskObjectsDef,
    ObjectSegmentations,
    PointCloudObjectDef,
    SceneSetup,
    get_observations_id_from_transforms_path,
    runtime_start,
    runtime_stop,
    runtime_pause,
    runtime_resume,
)

from clutt3rseg.clip_backends.duoduo import DEFAULT_DUODUO_CHECKPOINT, load_duoduo_clip
from clutt3rseg.deg_adapter import is_deg_dataset, materialize_workspace
from clutt3rseg.deg_postprocess import apply_deg_filters, build_instance_masks_from_leaves
from clutt3rseg.initial_segmenter import (
    initial_segmentation_consistency,
    validate_initial_dataset,
)
from clutt3rseg.mask_backends.grounded_sam import generate_masks_for_frames, load_grounded_sam
from clutt3rseg.tree_artifacts import ARTIFACT_NAME, load_instance_tree_artifact
from clutt3rseg.tree_builder import build_instance_tree_artifact, resolve_variant, write_instance_tree_artifact


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

    workspace_dir: Optional[Path] = None
    """For deg datasets: where to materialize the Clutt3R-Seg workspace (images/depth/masks/
    tree). Defaults to a stable per-dataset dir under the system temp, so masks/tree cache
    across runs. Ignored for the native clutt3r sample layout."""

    refresh_workspace: bool = False
    """For deg datasets: rebuild the workspace from scratch (re-export RGB/depth, drop cached
    masks and tree) instead of reusing it."""

    initial_idx: Optional[str] = None
    """Comma/space separated initial frame indices. Defaults to instance_tree.json's initial_idx.
    Required when no instance_tree.json exists and it has to be built."""

    build_tree_if_missing: bool = True
    """If data/instance_tree.json is absent, reconstruct it with the paper's tree builder
    (clutt3rseg.tree_builder) instead of failing. Requires --initial-idx."""

    tree_variant: Literal["paper", "improved"] = "paper"
    """Grouping variant used when auto-building the tree: 'paper' (default, faithful method
    section: weighted-Jaccard spatial, average linkage) or 'improved' (overlap coefficient,
    higher recall on partial cross-view masks)."""

    update_idx: Optional[str] = None
    """When building the tree, also emit containment trees for these update frames."""

    generate_masks_if_missing: bool = True
    """If data/instance_masks/ is absent, generate them with Grounded-SAM (GroundingDINO +
    SAM, prompt below) instead of failing. The release ships no detector."""

    mask_prompt: str = "object"
    """Grounded-SAM text prompt for mask generation (the paper uses the single token 'object')."""

    mask_box_threshold: float = 0.20
    """GroundingDINO box confidence threshold (0.20 calibrated to reproduce the sample masks)."""

    mask_text_threshold: float = 0.20
    """GroundingDINO text confidence threshold for mask generation."""

    target_prompt: str = "object"
    """Language prompt for the upstream target export. The deg glue exports all instances regardless."""

    voxel_size: Optional[float] = None
    """Point/super-voxel resolution for the spatial substrate. Defaults to 0.004 m
    for deg datasets - matching the point resolution the other deg baselines use
    (SAM3D's 0.0035 m voxelization, the 0.004 m TSDF voxel in MaskClustering /
    Open3DIS / SAI3D) - and 0.005 m for the native clutt3r samples (keeps the
    shipped-tree reproduction exact). NB this is a resolution, not SAM3D's 0.02 m
    cross-view *matching* tolerance: clutt3r associates views by shared super-voxels
    (no separate mutual-NN step), so coarsening to 0.02 m over-merges distinct
    objects on clean scenes without fixing under-association on noisy ones."""

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

    deg_filters: bool = True
    """Apply the deg-harness scene filters including workspace crop (table-plane voxel grid),
    keep-if-seen-in >= N frames, and table-instance removal. Needs the deg SceneSetup
    from scene_path; auto-skipped for the native clutt3r samples (no SceneSetup)."""

    min_frame_count: int = 3
    """Drop instances observed in fewer than this many frames (deg_filters)."""

    workspace_voxel_size: float = 0.02
    """Voxel size of the table-plane workspace grid used to crop instances (deg_filters);
    matches the 0.02 m grid the other baselines use."""

    remove_table: bool = True
    """Within deg_filters, drop the instance lying on the table plane."""

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


def _load_scene(scene_path: Optional[Path], logger: logging.Logger) -> Optional[SceneSetup]:
    """Load the deg SceneSetup pickle for the parity filters, or None if absent/unusable."""
    if scene_path is None or not Path(scene_path).exists():
        return None
    try:
        scene = SceneSetup.load(scene_path)
    except Exception as exc:  # noqa: BLE001 - any unpickle failure means we just skip the filters
        logger.warning("Could not load SceneSetup from %s: %s", scene_path, exc)
        return None
    if getattr(scene, "ground_gaussians", None) is None or getattr(scene, "ground_plane", None) is None:
        logger.warning("SceneSetup at %s has no ground_gaussians/ground_plane; skipping parity filters.", scene_path)
        return None
    return scene


def _generate_masks(experiment_data_dir: Path, args: "Args", device: str, logger: logging.Logger, rt: "dict | None" = None) -> None:
    """Generate Grounded-SAM masks into data/instance_masks/ for the frames we will use."""
    import json

    data_dir = experiment_data_dir / "data"
    frames_meta = json.loads((data_dir / "transforms.json").read_text())["frames"]
    if args.initial_idx is not None:
        idx = set(_parse_initial_idx(args.initial_idx))
        if args.update_idx:
            idx |= set(_parse_initial_idx(args.update_idx))
        idx = sorted(idx)
    else:
        idx = list(range(len(frames_meta)))
    image_paths = {i: data_dir / Path(frames_meta[i]["file_path"]).with_suffix(".png") for i in idx}

    logger.info("No instance_masks/ found; generating Grounded-SAM masks (prompt=%r) for %d frames ...", args.mask_prompt, len(idx))
    if rt is not None:
        runtime_pause(rt)  # exclude Grounded-SAM checkpoint load from the timed compute
    gsam = load_grounded_sam(device=device)
    if rt is not None:
        runtime_resume(rt)
    counts = generate_masks_for_frames(
        gsam,
        image_paths,
        data_dir / "instance_masks",
        prompt=args.mask_prompt,
        box_threshold=args.mask_box_threshold,
        text_threshold=args.mask_text_threshold,
    )
    logger.info("Generated %d masks over %d frames", sum(counts.values()), len(counts))


def run() -> None:
    logger = logging.getLogger("clutt3rseg-segmenter")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)-18s: %(levelname)-8s %(message)s"))
    logger.addHandler(handler)
    # Surface the tree builder's, mask backend's and deg-filter logs.
    for aux in ("clutt3rseg-builder", "clutt3rseg-gsam", "clutt3rseg-degfilter"):
        aux_logger = logging.getLogger(aux)
        aux_logger.setLevel(logging.INFO)
        aux_logger.addHandler(handler)

    args = tyro.cli(Args)

    transforms_path = args.transforms_path.resolve()

    # All upstream chatter goes to stderr; only the final objects_path line is on stdout.
    with contextlib.redirect_stdout(sys.stderr):
        logger.info("Starting Clutt3R-Seg initialization")
        logger.info("transforms_path=%s", transforms_path)

        # A deg dataset (per-frame K, OpenGL poses, depth_path) is adapted into a
        # temp Clutt3R-Seg `data/` workspace; the bundled clutt3r samples are used
        # in place. See clutt3rseg.deg_adapter.
        is_deg = is_deg_dataset(transforms_path)

        # deg datasets use 0.004 m to match the other baselines' point resolution
        # (SAM3D 0.0035, mesh TSDF 0.004); native samples keep 0.005 m so the
        # shipped-tree reproduction stays exact. See Args.voxel_size.
        if args.voxel_size is None:
            args.voxel_size = 0.004 if is_deg else 0.005
            logger.info("voxel_size defaulted to %.4f m (%s).", args.voxel_size, "deg" if is_deg else "native sample")

        if is_deg:
            obs_id = get_observations_id_from_transforms_path(transforms_path)
            workspace = (args.workspace_dir or Path(tempfile.gettempdir()) / "clutt3rseg_workspaces") / obs_id
            if args.refresh_workspace:
                shutil.rmtree(workspace, ignore_errors=True)
            logger.info("deg dataset detected; materializing workspace at %s ...", workspace)
            n_frames = materialize_workspace(transforms_path, workspace, overwrite_images=args.refresh_workspace)
            experiment_data_dir = workspace
            if args.initial_idx is None:
                args.initial_idx = ",".join(str(i) for i in range(n_frames))
                logger.info("No --initial-idx given; using all %d frames.", n_frames)
        else:
            experiment_data_dir = _experiment_dir_from_transforms(transforms_path)

        logger.info("experiment_data_dir=%s", experiment_data_dir)
        (experiment_data_dir / "output").mkdir(parents=True, exist_ok=True)
        tree_path = experiment_data_dir / "data" / ARTIFACT_NAME

        _rt = runtime_start("clutt3r-seg", scene=experiment_data_dir.name)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # The public release ships no Grounded-SAM detector; generate the per-frame
        # instance masks faithfully (GroundingDINO + SAM, prompt "object") when they
        # are absent, so the pipeline can run on sequences that ship only RGB-D.
        mask_dir = experiment_data_dir / "data" / "instance_masks"
        if args.generate_masks_if_missing and not (mask_dir.exists() and any(mask_dir.glob("mask_*.png"))):
            _generate_masks(experiment_data_dir, args, device, logger, rt=_rt)

        logger.info("Loading DuoduoCLIP (device=%s) ...", device)
        runtime_pause(_rt)  # exclude DuoduoCLIP checkpoint load from the timed compute
        clip = load_duoduo_clip(
            checkpoint=args.clip_checkpoint,
            device=device,
            duoduo_root=args.duoduo_root,
        )
        runtime_resume(_rt)

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
            logger.info(
                "No instance_tree.json found; building it (variant=%s) for initial_idx=%s ...",
                args.tree_variant, build_idx,
            )
            artifact = build_instance_tree_artifact(
                experiment_data_dir,
                build_idx,
                clip=clip,
                update_idx=update_idx,
                config=resolve_variant(args.tree_variant),
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
        node2inst = initial_data["node2inst"]
        logger.info("Recovered %d instances", len(inst2all_points))

        # Parity filters shared by the other deg baselines (workspace crop, min-frame,
        # table removal). They need the deg SceneSetup and only make sense for deg
        # scenes; the native clutt3r samples have no SceneSetup, so they are skipped.
        if args.deg_filters and is_deg:
            scene = _load_scene(args.scene_path, logger)
            if scene is not None:
                # Match the other baselines: the usual >=3 min-frame rule demands the
                # object appear in *every* frame when there are only 3 views, which is
                # too strict, so relax to >=2 when there are <=3 views.
                min_frames = 2 if len(initial_idx) <= 3 else args.min_frame_count
                inst2all_points, table_id = apply_deg_filters(
                    inst2all_points,
                    node2inst,
                    scene,
                    min_frames=min_frames,
                    workspace_voxel_size=args.workspace_voxel_size,
                    remove_table=args.remove_table,
                )
                logger.info("After deg filters: %d instances (table_id=%d)", len(inst2all_points), table_id)
            else:
                logger.warning("deg_filters requested but no usable SceneSetup at %s; skipping parity filters.", args.scene_path)

        n_objects = len(inst2all_points)
        if is_deg:
            # The deg harness (run_modelling) requires per-frame InstanceMaskObjectsDef,
            # as every other deg baseline returns. Build it from the surviving instances'
            # leaf masks (clutt3r's canonical 2D labeling); the ScanNet 3D eval reprojects
            # these onto the mesh when mesh_vertex_instance_ids is absent (as for
            # SAM3D / MaskClustering).
            import json

            ws_frames = json.loads((experiment_data_dir / "data" / "transforms.json").read_text())["frames"]
            orig_frames = json.loads(transforms_path.read_text()).get("frames", [])
            frame_specs = [
                (
                    i,
                    int(orig_frames[i]["id"]) if i < len(orig_frames) and "id" in orig_frames[i] else i,
                    int(fm["h"]),
                    int(fm["w"]),
                )
                for i, fm in enumerate(ws_frames)
            ]
            frame_ids, pixel_object_ids = build_instance_masks_from_leaves(node2inst, set(inst2all_points.keys()), mask_dir, frame_specs)
            objects = ObjectSegmentations(
                object_segmentations=InstanceMaskObjectsDef(frame_ids=frame_ids, pixel_object_ids=pixel_object_ids)
            )
        else:
            objects = ObjectSegmentations(object_segmentations=_to_point_cloud_objects(inst2all_points, args.estimate_normals))

        runtime_stop(_rt)

        out_root = Path(__file__).parent / "outputs"
        out_dir = out_root / f"{time.strftime('%Y%m%d-%H%M%S')}_{experiment_data_dir.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        objects_path = out_dir / "objectsdef.pkl"
        objects.save(objects_path)
        logger.info("Saved %d objects (%s) to %s", n_objects, "instance masks" if is_deg else "point clouds", objects_path)

    print(f"objects_path: {objects_path}")


if __name__ == "__main__":
    run()

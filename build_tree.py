"""CLI to (re)build ``data/instance_tree.json`` for a Clutt3R-Seg sequence.

The public release omits the instance-tree builder; this reconstructs it from the
paper (see :mod:`clutt3rseg.tree_builder`). It needs the same inputs the consumer
needs except the tree itself: ``data/transforms.json``, ``data/images``,
``data/depth`` (dense), and ``data/instance_masks`` (Grounded-SAM, prompt
"object").

Usage::

    python build_tree.py <experiment_data_dir> \
        --initial-idx 0,1,2,3,4,5,6,7 [--update-idx 8,9]

Writes ``<experiment_data_dir>/data/instance_tree.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Literal, Optional

import tyro

from clutt3rseg.clip_backends.duoduo import DEFAULT_DUODUO_CHECKPOINT, load_duoduo_clip
from clutt3rseg.tree_builder import (
    build_instance_tree_artifact,
    resolve_variant,
    write_instance_tree_artifact,
)


def _parse_idx(value: str) -> list[int]:
    return [int(tok) for tok in value.replace(",", " ").split() if tok]


@dataclass
class Args:
    experiment_data_dir: tyro.conf.Positional[Path]
    """Sequence directory containing data/transforms.json, images, depth, instance_masks."""

    variant: Literal["paper", "improved"] = "paper"
    """Grouping variant. 'paper' (default) = faithful method section: weighted-Jaccard
    spatial (tau_spat=0.5), semantic tau_sem=0.65, average linkage. 'improved' = overlap
    coefficient at tau_spat=0.4 (higher recall on partial cross-view masks). Override
    individual knobs below."""

    initial_idx: str = "0,1,2,3,4,5,6,7"
    """Comma/space separated initial frame indices to associate across views."""

    update_idx: Optional[str] = None
    """Optional comma/space separated update frame indices to emit containment trees for."""

    no_clip: bool = False
    """Skip DuoduoCLIP and run spatial-only grouping (no semantic stage)."""

    voxel_size: float = 0.005
    depth_scale: float = 0.001
    max_depth: float = 5.0
    gc_lambda: float = 0.010
    min_points: int = 3

    # Per-knob overrides (default None -> take the value from --variant).
    spatial_metric: Optional[Literal["overlap", "jaccard"]] = None
    tau_spat: Optional[float] = None
    tau_sem: Optional[float] = None
    linkage: Optional[Literal["average", "max"]] = None
    containment_thresh: Optional[float] = None

    duoduo_root: Optional[Path] = None
    """External DuoduoCLIP checkout. Defaults to $DUODUOCLIP_ROOT."""

    clip_checkpoint: str = DEFAULT_DUODUO_CHECKPOINT


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)-18s: %(levelname)-8s %(message)s")
    args = tyro.cli(Args)

    overrides = {
        k: v
        for k, v in {
            "spatial_metric": args.spatial_metric,
            "tau_spat": args.tau_spat,
            "tau_sem": args.tau_sem,
            "linkage": args.linkage,
            "containment_thresh": args.containment_thresh,
        }.items()
        if v is not None
    }
    config = replace(resolve_variant(args.variant), **overrides)

    clip = None
    if not args.no_clip:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip = load_duoduo_clip(checkpoint=args.clip_checkpoint, device=device, duoduo_root=args.duoduo_root)

    artifact = build_instance_tree_artifact(
        args.experiment_data_dir,
        _parse_idx(args.initial_idx),
        clip=clip,
        update_idx=_parse_idx(args.update_idx) if args.update_idx else None,
        config=config,
        voxel_size=args.voxel_size,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        gc_lambda=args.gc_lambda,
        min_points=args.min_points,
    )
    out_path = write_instance_tree_artifact(args.experiment_data_dir, artifact)
    n_leaves = len(artifact["initial"]["leaf2inst"])
    n_inst = len({e["instance"] for e in artifact["initial"]["leaf2inst"]})
    print(f"instance_tree: {out_path} ({n_leaves} leaves -> {n_inst} instances, {len(artifact['updates'])} update frames)")


if __name__ == "__main__":
    run()

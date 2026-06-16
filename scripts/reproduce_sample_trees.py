"""Reproduce the bundled instance_tree.json artifacts from the raw sample data.

The public release ships ``data/instance_tree.json`` for each sample but not the
builder that produced it. This script rebuilds each sample's tree from its
``transforms.json`` + ``depth`` + ``instance_masks`` using
:mod:`clutt3rseg.tree_builder` and reports how closely the reconstruction matches
the shipped artifact, for one or more grouping variants.

It is the confirmation that the reimplemented builder is correct: a high
agreement with the shipped trees means the reconstruction recovers the same
cross-view instances the authors released.

Metrics (initial segmentation, on leaves present in both trees):
  * ARI         -- Adjusted Rand Index of the two groupings.
  * IoU         -- mean best-match instance IoU (and how many match perfectly).
  * #inst       -- instance counts (built vs shipped).
  * leaf-cov    -- fraction of shipped leaves the builder also kept as leaves.

For update frames it compares the per-frame containment trees (leaf-set Jaccard,
leaf and edge counts), which depend only on the 2D containment routine.

Usage::

    pixi run --frozen python scripts/reproduce_sample_trees.py
    pixi run --frozen python scripts/reproduce_sample_trees.py --variants improved,paper --csv-out /tmp/repro.csv
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

from clutt3rseg.clip_backends.duoduo import DEFAULT_DUODUO_CHECKPOINT, load_duoduo_clip
from clutt3rseg.scene_substrate import build_scene_substrate
from clutt3rseg.tree_artifacts import ARTIFACT_NAME, leaf_id
from clutt3rseg.tree_builder import (
    _load_frame_masks,
    build_containment_forest,
    build_initial_leaf2inst,
    compute_leaf_embeddings,
    resolve_variant,
)


@dataclass
class Args:
    samples_root: Path = Path("samples")
    """Root directory of sample sequences."""

    sequences: Optional[str] = None
    """Comma separated sequence names. Default: every sub-dir with data/instance_tree.json."""

    variants: str = "improved,paper"
    """Comma separated grouping variants to evaluate."""

    csv_out: Optional[Path] = None
    """Optional path to write the per-(sequence, variant) metrics as CSV."""

    no_clip: bool = False
    """Skip DuoduoCLIP (spatial-only grouping; semantic stage disabled)."""

    voxel_size: float = 0.005
    depth_scale: float = 0.001
    max_depth: float = 5.0
    gc_lambda: float = 0.010
    min_points: int = 3

    duoduo_root: Optional[Path] = None
    clip_checkpoint: str = DEFAULT_DUODUO_CHECKPOINT


def _c2(x: int) -> int:
    return x * (x - 1) // 2


def _ari(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    pair = Counter(zip(a, b))
    ca = Counter(a)
    cb = Counter(b)
    idx = sum(_c2(v) for v in pair.values())
    ea = sum(_c2(v) for v in ca.values())
    eb = sum(_c2(v) for v in cb.values())
    exp = ea * eb / _c2(n)
    mx = 0.5 * (ea + eb)
    return (idx - exp) / (mx - exp) if mx != exp else 1.0


def _shipped_leaf2inst(artifact: dict) -> dict[tuple[int, int], int]:
    return {(e["frame"], e["mask"]): e["instance"] for e in artifact["initial"]["leaf2inst"]}


def _initial_metrics(built: dict[tuple[int, int], int], shipped: dict[tuple[int, int], int]) -> dict:
    common = sorted(set(built) & set(shipped))
    ari = _ari([shipped[k] for k in common], [built[k] for k in common]) if common else 0.0

    bi: dict[int, set] = defaultdict(set)
    si: dict[int, set] = defaultdict(set)
    for k in common:
        bi[built[k]].add(k)
        si[shipped[k]].add(k)
    ious = [max((len(bi[x] & si[y]) / len(bi[x] | si[y]) for y in si), default=0.0) for x in bi]

    return {
        "leaves_built": len(built),
        "leaves_shipped": len(shipped),
        "common": len(common),
        "leaf_cov": len(common) / len(shipped) if shipped else 0.0,
        "inst_built": len(set(built.values())),
        "inst_shipped": len(set(shipped.values())),
        "ARI": ari,
        "IoU_mean": float(np.mean(ious)) if ious else 0.0,
        "IoU_perfect": int(sum(i > 0.999 for i in ious)),
    }


def _update_metrics(data_dir: Path, artifact: dict, containment_thresh: float) -> list[dict]:
    rows = []
    for frame_str, upd in artifact.get("updates", {}).items():
        frame = int(frame_str)
        masks = _load_frame_masks(data_dir, frame)
        if not masks:
            continue
        parent_of, _children, leaves, _desc = build_containment_forest(masks, containment_thresh)
        built_leaves = {leaf_id(frame, m) for m in leaves}
        shipped_leaves = {leaf_id(ln["frame"], ln["mask"]) for ln in upd.get("leaf_nodes", [])}
        inter = len(built_leaves & shipped_leaves)
        union = len(built_leaves | shipped_leaves)
        rows.append({
            "frame": frame,
            "leaves_built": len(built_leaves),
            "leaves_shipped": len(shipped_leaves),
            "leaf_jaccard": inter / union if union else 1.0,
            "edges_built": len(parent_of),
            "edges_shipped": len(upd.get("parent_of", {})),
        })
    return rows


def run() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(name)-18s: %(levelname)-8s %(message)s")
    args = tyro.cli(Args)

    if args.sequences:
        seqs = [args.samples_root / s.strip() for s in args.sequences.split(",") if s.strip()]
    else:
        seqs = sorted(p for p in args.samples_root.iterdir() if (p / "data" / ARTIFACT_NAME).exists())

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    configs = {v: resolve_variant(v) for v in variants}

    clip = None
    if not args.no_clip:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip = load_duoduo_clip(checkpoint=args.clip_checkpoint, device=device, duoduo_root=args.duoduo_root)

    header = f"{'sequence':16s} {'variant':9s} {'leaves(b/s)':12s} {'inst(b/s)':10s} {'ARI':>6s} {'IoU':>6s} {'perfect':>8s} {'leaf-cov':>8s}"
    print(header)
    print("-" * len(header))

    csv_rows: list[dict] = []
    for seq in seqs:
        shipped = json.loads((seq / "data" / ARTIFACT_NAME).read_text())
        ship_l2i = _shipped_leaf2inst(shipped)
        initial_idx = [int(i) for i in shipped["initial"]["initial_idx"]]

        substrate = build_scene_substrate(
            seq, initial_idx,
            voxel_size=args.voxel_size, depth_scale=args.depth_scale,
            max_depth=args.max_depth, gc_lambda=args.gc_lambda, min_points=args.min_points,
        )
        embeddings = compute_leaf_embeddings(substrate.crops, clip) if clip is not None else {}

        for v in variants:
            built_l2i, _forests, _ground = build_initial_leaf2inst(substrate, embeddings, configs[v])
            m = _initial_metrics(built_l2i, ship_l2i)
            leaves_col = f"{m['leaves_built']}/{m['leaves_shipped']}"
            inst_col = f"{m['inst_built']}/{m['inst_shipped']}"
            perfect_col = f"{m['IoU_perfect']}/{m['inst_built']}"
            print(
                f"{seq.name:16s} {v:9s} {leaves_col:12s} {inst_col:10s} "
                f"{m['ARI']:6.3f} {m['IoU_mean']:6.3f} {perfect_col:>8s} {m['leaf_cov']:8.3f}"
            )
            csv_rows.append({"sequence": seq.name, "variant": v, **m})

        upd_rows = _update_metrics(seq / "data", shipped, configs[variants[0]].containment_thresh)
        for ur in upd_rows:
            print(
                f"  update frame {ur['frame']}: leaves(b/s)={ur['leaves_built']}/{ur['leaves_shipped']} "
                f"leaf-Jaccard={ur['leaf_jaccard']:.3f} edges(b/s)={ur['edges_built']}/{ur['edges_shipped']}"
            )

    if args.csv_out and csv_rows:
        import csv

        with args.csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    run()

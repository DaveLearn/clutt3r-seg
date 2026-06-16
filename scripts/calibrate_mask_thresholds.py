"""Calibrate Grounded-SAM thresholds to reproduce the bundled samples' masks.

The release ships the authors' Grounded-SAM ``instance_masks/`` but not the
thresholds used. This sweeps GroundingDINO's box/text threshold and scores how
well our Grounded-SAM reproduces the shipped masks on the sample sequences, so we
can pick a faithful default.

For each threshold we match generated masks to shipped masks per frame by IoU and
report:
  * recall  -- mean over shipped masks of the best IoU with a generated mask
               (how well we recover the authors' masks),
  * precision-- mean over generated masks of the best IoU with a shipped mask,
  * F1       -- harmonic mean of the two,
  * count    -- generated/shipped mask totals.

Usage::

    pixi run --frozen python scripts/calibrate_mask_thresholds.py \
        --sequences sample_seq2,sample_seq4 --frames 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path

import cv2
import numpy as np
import tyro

from clutt3rseg.mask_backends.grounded_sam import DEFAULT_PROMPT, generate_instance_masks, load_grounded_sam


@dataclass
class Args:
    samples_root: Path = Path("samples")
    sequences: str = "sample_seq2,sample_seq4"
    frames: str = "0,1,2,3,4,5,6,7"
    thresholds: str = "0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30"
    prompt: str = DEFAULT_PROMPT


def _load_shipped(mask_dir: Path, frame: int) -> list[np.ndarray]:
    out = []
    for p in sorted(glob.glob(str(mask_dir / f"mask_{frame:06d}_*.png"))):
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            out.append(m > 0)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_and(a, a).sum() + np.logical_and(b, b).sum() - inter
    return float(inter) / float(union) if union else 0.0


def _best_ious(src: list[np.ndarray], ref: list[np.ndarray]) -> list[float]:
    return [max((_iou(s, r) for r in ref), default=0.0) for s in src]


def run() -> None:
    import torch  # noqa

    args = tyro.cli(Args)
    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    frames = [int(t) for t in args.frames.replace(",", " ").split() if t]
    thresholds = [float(t) for t in args.thresholds.split(",") if t]

    from PIL import Image

    gsam = load_grounded_sam(device="cuda")

    # Cache generated masks per (seq, frame, threshold) so we encode once per setting.
    print(f"{'threshold':>9s} {'gen/ship':>10s} {'recall':>7s} {'prec':>6s} {'F1':>6s}")
    print("-" * 44)
    best = (None, -1.0)
    for t in thresholds:
        n_gen = n_ship = 0
        recalls: list[float] = []
        precs: list[float] = []
        for seq in seqs:
            data = args.samples_root / seq / "data"
            mask_dir = data / "instance_masks"
            for f in frames:
                shipped = _load_shipped(mask_dir, f)
                if not shipped:
                    continue
                img_paths = glob.glob(str(data / "images" / f"*{f:06d}*"))
                if not img_paths:
                    continue
                gen = generate_instance_masks(gsam, Image.open(img_paths[0]), args.prompt, box_threshold=t, text_threshold=t)
                n_gen += len(gen)
                n_ship += len(shipped)
                if gen:
                    precs += _best_ious(gen, shipped)
                recalls += _best_ious(shipped, gen)
        recall = float(np.mean(recalls)) if recalls else 0.0
        prec = float(np.mean(precs)) if precs else 0.0
        f1 = 2 * recall * prec / (recall + prec) if (recall + prec) else 0.0
        print(f"{t:9.2f} {f'{n_gen}/{n_ship}':>10s} {recall:7.3f} {prec:6.3f} {f1:6.3f}")
        if f1 > best[1]:
            best = (t, f1)
    print(f"\nBest threshold by F1: {best[0]} (F1={best[1]:.3f})")


if __name__ == "__main__":
    run()

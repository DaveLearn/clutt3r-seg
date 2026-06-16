"""CLI to generate Grounded-SAM instance masks for a Clutt3R-Seg sequence.

The public release expects ``data/instance_masks/mask_<frame>_<inst>.png`` to
already exist; this produces them faithfully with Grounded-SAM (GroundingDINO +
SAM, prompt "object"). See :mod:`clutt3rseg.mask_backends.grounded_sam`.

Usage::

    python generate_masks.py <experiment_data_dir> [--frames 0,1,2,...] [--overwrite]

Reads RGB from ``<experiment_data_dir>/data/<frame.file_path>`` and writes masks
to ``<experiment_data_dir>/data/instance_masks/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Optional

import tyro

from clutt3rseg.mask_backends.grounded_sam import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_DINO_ID,
    DEFAULT_PROMPT,
    DEFAULT_SAM_ID,
    DEFAULT_TEXT_THRESHOLD,
    generate_masks_for_frames,
    load_grounded_sam,
)


@dataclass
class Args:
    experiment_data_dir: tyro.conf.Positional[Path]
    """Sequence directory containing data/transforms.json and the RGB images."""

    frames: Optional[str] = None
    """Comma/space separated frame indices. Default: all frames in transforms.json."""

    prompt: str = DEFAULT_PROMPT
    box_threshold: float = DEFAULT_BOX_THRESHOLD
    text_threshold: float = DEFAULT_TEXT_THRESHOLD
    overwrite: bool = False

    dino_id: str = DEFAULT_DINO_ID
    sam_id: str = DEFAULT_SAM_ID


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)-18s: %(levelname)-8s %(message)s")
    args = tyro.cli(Args)

    import torch

    data_dir = args.experiment_data_dir / "data"
    meta = json.loads((data_dir / "transforms.json").read_text())
    frames_meta = meta["frames"]

    if args.frames is not None:
        idx = [int(t) for t in args.frames.replace(",", " ").split() if t]
    else:
        idx = list(range(len(frames_meta)))

    image_paths = {i: data_dir / Path(frames_meta[i]["file_path"]).with_suffix(".png") for i in idx}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gsam = load_grounded_sam(device=device, dino_id=args.dino_id, sam_id=args.sam_id)
    counts = generate_masks_for_frames(
        gsam,
        image_paths,
        data_dir / "instance_masks",
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        overwrite=args.overwrite,
    )
    print(f"instance_masks: {data_dir / 'instance_masks'} ({sum(counts.values())} masks over {len(counts)} frames)")


if __name__ == "__main__":
    run()

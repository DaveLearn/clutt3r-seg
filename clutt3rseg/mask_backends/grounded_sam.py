"""Grounded-SAM 2D instance-mask generation (the paper's mask front-end).

Clutt3R-Seg prompts **Grounded SAM** with the single token "object" to produce
per-frame 2D masks ("yielding proper segmentations, under-segmentations, and
over-segmentations from a single pass"). The public release expects those masks
to already exist on disk; this module reconstructs that step.

Grounded-SAM = GroundingDINO (open-vocabulary box detection from a text prompt)
followed by SAM (box-prompted segmentation). We run both via HuggingFace
``transformers`` (``IDEA-Research/grounding-dino-*`` + ``facebook/sam-vit-*``),
which uses the same model weights as the original Grounded-SAM repo but without
its custom CUDA build. Checkpoints download from the HF Hub on first use.

Masks are written as ``mask_<frame:06d>_<inst:02d>.png`` (0/255, 1-indexed),
exactly the convention the rest of the pipeline reads
(:func:`clutt3rseg.utils.refine_masks_by_depth`).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clutt3rseg-gsam")

DEFAULT_DINO_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM_ID = "facebook/sam-vit-huge"
DEFAULT_PROMPT = "object"
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.25


@dataclass
class GroundedSAM:
    dino_processor: object
    dino_model: object
    sam_processor: object
    sam_model: object
    device: str


def load_grounded_sam(
    device: str = "cuda",
    dino_id: str = DEFAULT_DINO_ID,
    sam_id: str = DEFAULT_SAM_ID,
) -> GroundedSAM:
    """Load GroundingDINO + SAM from HuggingFace (downloads checkpoints on first use)."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

    logger.info("Loading GroundingDINO (%s) ...", dino_id)
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()

    logger.info("Loading SAM (%s) ...", sam_id)
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(device).eval()

    return GroundedSAM(dino_processor, dino_model, sam_processor, sam_model, device)


def _normalize_prompt(prompt: str) -> str:
    # GroundingDINO expects a lowercase, '.'-terminated caption.
    prompt = prompt.strip().lower()
    return prompt if prompt.endswith(".") else prompt + "."


def generate_instance_masks(
    gsam: GroundedSAM,
    image: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    *,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    min_area: int = 50,
) -> list[np.ndarray]:
    """Detect boxes for ``prompt`` with GroundingDINO, segment each with SAM.

    Returns a list of boolean HxW masks (one per detected object), largest first.
    """
    import torch

    image = image.convert("RGB")
    w, h = image.size

    dino_inputs = gsam.dino_processor(images=image, text=_normalize_prompt(prompt), return_tensors="pt").to(gsam.device)
    with torch.no_grad():
        dino_outputs = gsam.dino_model(**dino_inputs)
    # The box-confidence kwarg was renamed box_threshold -> threshold in transformers 5.x.
    try:
        detections = gsam.dino_processor.post_process_grounded_object_detection(
            dino_outputs, dino_inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(h, w)],
        )[0]
    except TypeError:
        detections = gsam.dino_processor.post_process_grounded_object_detection(
            dino_outputs, dino_inputs["input_ids"], box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(h, w)],
        )[0]
    boxes = detections["boxes"]
    if boxes.shape[0] == 0:
        return []

    sam_inputs = gsam.sam_processor(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(gsam.device)
    with torch.no_grad():
        sam_outputs = gsam.sam_model(**sam_inputs)
    masks = gsam.sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]  # (N, 3, H, W): SAM returns 3 candidate masks per box
    scores = sam_outputs.iou_scores.cpu()[0]  # (N, 3)
    best = scores.argmax(dim=1)

    out: list[np.ndarray] = []
    for i in range(masks.shape[0]):
        m = masks[i, int(best[i])].numpy().astype(bool)
        if int(m.sum()) >= min_area:
            out.append(m)
    out.sort(key=lambda m: int(m.sum()), reverse=True)
    return out


def write_frame_masks(out_dir: Path, frame_idx: int, masks: list[np.ndarray]) -> int:
    """Write masks for one frame as ``mask_<frame:06d>_<inst:02d>.png`` (1-indexed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(masks, start=1):
        cv2.imwrite(str(out_dir / f"mask_{frame_idx:06d}_{k:02d}.png"), (m.astype(np.uint8) * 255))
    return len(masks)


def generate_masks_for_frames(
    gsam: GroundedSAM,
    image_paths: dict[int, Path],
    out_dir: Path,
    prompt: str = DEFAULT_PROMPT,
    *,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    overwrite: bool = False,
) -> dict[int, int]:
    """Generate + write masks for ``{frame_idx: rgb_path}``; returns {frame_idx: n_masks}."""
    counts: dict[int, int] = {}
    for frame_idx, rgb_path in sorted(image_paths.items()):
        if not overwrite and list(out_dir.glob(f"mask_{frame_idx:06d}_*.png")):
            counts[frame_idx] = len(list(out_dir.glob(f"mask_{frame_idx:06d}_*.png")))
            continue
        image = Image.open(rgb_path)
        masks = generate_instance_masks(
            gsam, image, prompt, box_threshold=box_threshold, text_threshold=text_threshold
        )
        counts[frame_idx] = write_frame_masks(out_dir, frame_idx, masks)
        logger.info("Frame %d: %d masks", frame_idx, counts[frame_idx])
    return counts

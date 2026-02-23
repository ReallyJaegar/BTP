# =============================================================================
# segmenter.py — Step 3: Generate binary masks for each selected object
# =============================================================================
# Uses SAM (Segment Anything Model) in automatic mode to generate all masks
# in the image, then assigns one mask per LLM-selected label.
#
# Mask → Label assignment strategy (weakest link — see note below):
#   Currently uses SAM's predicted IoU score as a proxy for confidence,
#   assigning the highest-scored unused mask to each label in order.
#
# ── TODO for production ───────────────────────────────────────────────────────
#   Replace the assignment step with CLIP-based matching:
#     1. Crop each mask region from the image.
#     2. Embed both the crop and the label string with CLIP.
#     3. Assign the mask with the highest cosine similarity to each label.
#   Alternatively, use SAM3's text-prompted mode (facebook/sam3) which
#   accepts a text query and returns masks directly.
# =============================================================================

import os
import numpy as np
from PIL import Image

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import config
from utils import save_mask


# ── Types ─────────────────────────────────────────────────────────────────────

# Each item in the returned list looks like:
# {
#   "label":     "bus",
#   "prompt":    "a Nigerian danfo bus ...",
#   "mask":      <PIL.Image grayscale>,
#   "mask_path": "outputs/mask_bus.png",
# }
SegmentedItem = dict


# ── Public API ────────────────────────────────────────────────────────────────

def generate_masks(
    image:              Image.Image,
    label_prompt_pairs: list[dict],
    output_dir:         str,
) -> list[SegmentedItem]:
    """
    For each (label, prompt) pair from the LLM, find the best matching
    SAM mask and return the full mask metadata.

    Args:
        image:              The original PIL RGB image.
        label_prompt_pairs: Output from llm_brain.select_and_generate_prompts().
        output_dir:         Directory to save mask PNG files.

    Returns:
        List of SegmentedItem dicts (label + prompt + mask PIL image + path).
    """
    image_np  = np.array(image)
    all_masks = _run_sam(image_np)

    filtered  = _filter_masks_by_area(all_masks)
    print(f"  SAM total masks: {len(all_masks)}  →  after area filter: {len(filtered)}")

    assigned  = _assign_masks_to_labels(filtered, label_prompt_pairs, image_np, output_dir)
    return assigned


# ── Private helpers ───────────────────────────────────────────────────────────

def _run_sam(image_np: np.ndarray) -> list[dict]:
    """Load SAM and generate all masks for the image."""
    print(f"  Loading SAM ({config.SAM_MODEL_TYPE})...")
    sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT)
    sam.to(config.DEVICE)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=config.SAM_PRED_IOU_THRESH,
        stability_score_thresh=config.SAM_STABILITY_SCORE_THRESH,
    )
    return mask_generator.generate(image_np)


def _filter_masks_by_area(masks: list[dict]) -> list[dict]:
    """
    Remove masks that are too small (noise) or too large (background).
    Thresholds are controlled by SAM_AREA_MIN / SAM_AREA_MAX in config.
    """
    return [
        m for m in masks
        if config.SAM_AREA_MIN < m["area"] < config.SAM_AREA_MAX
    ]


def _assign_masks_to_labels(
    masks:              list[dict],
    label_prompt_pairs: list[dict],
    image_np:           np.ndarray,
    output_dir:         str,
) -> list[SegmentedItem]:
    """
    Assign one mask to each label.

    Current strategy: rank masks by predicted IoU, assign highest-scored
    unused mask to each label in order.

    See module docstring for the recommended CLIP-based upgrade path.
    """
    # Sort by SAM confidence score (descending)
    ranked = sorted(masks, key=lambda m: m["predicted_iou"], reverse=True)

    results:     list[SegmentedItem] = []
    used_indices: set[int]           = set()

    for pair in label_prompt_pairs:
        label  = pair["label"]
        prompt = pair["prompt"]

        # Find the highest-ranked mask not yet assigned
        mask_np = None
        for idx, m in enumerate(ranked):
            if idx not in used_indices:
                mask_np = m["segmentation"]
                used_indices.add(idx)
                break

        if mask_np is None:
            print(f"  Warning: no mask left for '{label}' — skipping.")
            continue

        mask_path = save_mask(mask_np, label, output_dir)
        mask_pil  = Image.open(mask_path)
        print(f"  Assigned mask for '{label}' → {mask_path}")

        results.append({
            "label":     label,
            "prompt":    prompt,
            "mask":      mask_pil,
            "mask_path": mask_path,
        })

    return results

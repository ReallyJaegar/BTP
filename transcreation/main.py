# =============================================================================
# main.py — Entry point for the Image Transcreation Pipeline
# =============================================================================
#
# Orchestrates the four-step pipeline:
#
#   Image
#     │
#     ├─ Step 1 ── tagger.py      Extract object tags + scene caption
#     │
#     ├─ Step 2 ── llm_brain.py   LLM selects salient objects + writes prompts
#     │
#     ├─ Step 3 ── segmenter.py   SAM generates a binary mask per object
#     │
#     └─ Step 4 ── inpainter.py   Iterative SD inpainting → final image
#
# All settings (paths, models, target culture) live in config.py.
# =============================================================================

import os
from PIL import Image

import config
from tagger    import extract_tags
from llm_brain import select_and_generate_prompts
from segmenter import generate_masks
from inpainter import run_iterative_inpainting
from utils     import overlay_masks, show_before_after, print_step


def run_pipeline(image_path: str | None = None) -> Image.Image:
    """
    Execute the full transcreation pipeline end-to-end.

    Args:
        image_path: Path to the input image. Defaults to config.IMAGE_PATH.

    Returns:
        The final transcreated PIL image.
    """
    image_path = image_path or config.IMAGE_PATH
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  IMAGE TRANSCREATION PIPELINE")
    print(f"  Input  : {image_path}")
    print(f"  Culture: {config.TARGET_CULTURE}")
    print("=" * 60)

    # ── Load image ────────────────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    print(f"\nLoaded image: {image.size[0]}×{image.size[1]} px")

    # ── Step 1: Tag extraction ────────────────────────────────────────────────
    print_step(1, "Tag Extraction")
    tags, caption = extract_tags(image)

    if not tags:
        print("No objects detected. Exiting.")
        return image

    # ── Step 2: LLM — select culturally salient objects + write prompts ───────
    print_step(2, "LLM Cultural Selection")
    label_prompt_pairs = select_and_generate_prompts(tags, caption)

    if not label_prompt_pairs:
        print("LLM found no culturally salient objects to replace. Exiting.")
        return image

    # ── Step 3: SAM segmentation ──────────────────────────────────────────────
    print_step(3, "SAM Segmentation")
    segmented_items = generate_masks(image, label_prompt_pairs, config.OUTPUT_DIR)

    if not segmented_items:
        print("No masks generated. Exiting.")
        return image

    # Save a visual overview of all masks overlaid on the image
    mask_overlay_path = os.path.join(config.OUTPUT_DIR, "mask_overlay.png")
    overlay = overlay_masks(image, [item["mask"] for item in segmented_items])
    overlay.save(mask_overlay_path)
    print(f"  Mask overlay saved → {mask_overlay_path}")

    # ── Step 4: Iterative inpainting ──────────────────────────────────────────
    print_step(4, "Iterative Inpainting")
    final_image = run_iterative_inpainting(image, segmented_items, config.OUTPUT_DIR)

    # ── Save & display final result ───────────────────────────────────────────
    final_path = os.path.join(config.OUTPUT_DIR, "final_transcreated.png")
    final_image.save(final_path)
    print(f"\n✓ Final image saved → {final_path}")

    comparison_path = os.path.join(config.OUTPUT_DIR, "before_after.png")
    show_before_after(image, final_image, save_path=comparison_path)

    return final_image


if __name__ == "__main__":
    run_pipeline()

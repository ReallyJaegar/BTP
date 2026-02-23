# =============================================================================
# inpainter.py — Step 4: Iterative Stable Diffusion inpainting
# =============================================================================
# Applies SD inpainting sequentially over each (mask, prompt) pair.
# The output of each inpainting step is used as the input to the next,
# so all edits accumulate into a single coherent final image.
#
# Intermediate results are saved to disk after each step so you can inspect
# the progression or restart from a specific step if something goes wrong.
# =============================================================================

import os
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

import config


# ── Public API ────────────────────────────────────────────────────────────────

def run_iterative_inpainting(
    image:               Image.Image,
    segmented_items:     list[dict],
    output_dir:          str,
) -> Image.Image:
    """
    Inpaint each masked region sequentially using Stable Diffusion.

    Args:
        image:           The original (or starting) PIL RGB image.
        segmented_items: Output from segmenter.generate_masks() —
                         list of dicts with keys: label, prompt, mask, mask_path.
        output_dir:      Directory to save intermediate and final images.

    Returns:
        The fully transcreated PIL image after all inpainting steps.
    """
    pipe = _load_pipeline()

    # SD inpainting requires fixed input dimensions
    target_size    = (config.INPAINT_SIZE, config.INPAINT_SIZE)
    current_image  = image.resize(target_size)

    for step, item in enumerate(segmented_items, start=1):
        label  = item["label"]
        prompt = item["prompt"]
        mask   = item["mask"].resize(target_size)

        print(f"  [{step}/{len(segmented_items)}] Inpainting '{label}'...")

        current_image = _inpaint_single(
            pipeline=pipe,
            image=current_image,
            mask=mask,
            prompt=prompt,
        )

        # Save intermediate result for debugging / inspection
        step_path = os.path.join(
            output_dir, f"step_{step:02d}_{label.replace(' ', '_')}.png"
        )
        current_image.save(step_path)
        print(f"    Intermediate result → {step_path}")

    return current_image


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_pipeline() -> StableDiffusionInpaintPipeline:
    """Load and return the Stable Diffusion inpainting pipeline."""
    print(f"  Loading inpainting model: {config.INPAINT_MODEL}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config.INPAINT_MODEL,
        torch_dtype=torch.float16,
    ).to("cuda")
    return pipe


def _inpaint_single(
    pipeline: StableDiffusionInpaintPipeline,
    image:    Image.Image,
    mask:     Image.Image,
    prompt:   str,
) -> Image.Image:
    """
    Run one inpainting pass and return the result image.

    Args:
        pipeline: Pre-loaded SD inpainting pipeline.
        image:    Current image (RGB PIL, resized to INPAINT_SIZE).
        mask:     Binary mask (grayscale PIL, same size as image).
        prompt:   Text prompt describing what to generate in the masked region.

    Returns:
        Inpainted PIL image.
    """
    result = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=config.GUIDANCE_SCALE,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        padding_mask_crop=config.PADDING_MASK_CROP,
    ).images[0]

    return result

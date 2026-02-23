# =============================================================================
# tagger.py — Step 1: Extract object tags from the input image
# =============================================================================
# Uses two models in combination:
#   • BLIP-2  → generates a natural-language caption of the whole scene
#   • OWL-ViT → open-vocabulary object detection against candidate labels
#
# The caption provides scene context for the LLM, while the detected labels
# give a structured list of objects present in the image.
# =============================================================================

import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    OwlViTProcessor,
    OwlViTForObjectDetection,
)

import config


# ── Public API ────────────────────────────────────────────────────────────────

def extract_tags(image: Image.Image) -> tuple[list[str], str]:
    """
    Detect objects and generate a caption for the given image.

    Args:
        image: A PIL RGB image.

    Returns:
        tags:    Sorted, deduplicated list of object label strings.
        caption: A natural-language description of the scene (from BLIP-2).
    """
    caption = _get_caption(image)
    detected_labels = _detect_objects(image)

    # Merge caption nouns + detected labels into one deduplicated set
    caption_nouns = _extract_nouns_from_caption(caption)
    all_tags = sorted(set(caption_nouns) | set(detected_labels))

    print(f"  Caption  : {caption}")
    print(f"  Tags     : {all_tags}")
    return all_tags, caption


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_caption(image: Image.Image) -> str:
    """Run BLIP-2 to produce a natural-language caption for the image."""
    print("  Loading BLIP-2...")
    processor = Blip2Processor.from_pretrained(config.BLIP2_MODEL)
    model     = Blip2ForConditionalGeneration.from_pretrained(
        config.BLIP2_MODEL, torch_dtype=torch.float16
    ).to(config.DEVICE)

    inputs  = processor(images=image, return_tensors="pt").to(config.DEVICE, torch.float16)
    out     = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


def _detect_objects(image: Image.Image) -> list[str]:
    """
    Run OWL-ViT open-vocabulary detection against the candidate label list
    defined in config.DETECTION_CANDIDATE_LABELS.
    """
    print("  Loading OWL-ViT...")
    processor = OwlViTProcessor.from_pretrained(config.OWLVIT_MODEL)
    model     = OwlViTForObjectDetection.from_pretrained(
        config.OWLVIT_MODEL
    ).to(config.DEVICE)

    inputs = processor(
        text=[config.DETECTION_CANDIDATE_LABELS],
        images=image,
        return_tensors="pt",
    ).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.1,
        target_sizes=[image.size[::-1]],
    )[0]

    detected = [
        config.DETECTION_CANDIDATE_LABELS[idx]
        for idx in results["labels"]
    ]
    return list(set(detected))


def _extract_nouns_from_caption(caption: str) -> list[str]:
    """
    Lightweight noun extraction from a caption string — no NLP library needed.
    Filters short/common words and returns a list of candidate nouns.
    """
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "is", "are", "was", "were",
        "this", "that", "there", "some", "many", "very", "quite",
    }
    words = [
        w.strip(".,!?;:\"'").lower()
        for w in caption.split()
    ]
    return [w for w in words if len(w) > 3 and w not in STOPWORDS]

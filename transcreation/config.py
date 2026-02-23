# =============================================================================
# config.py — All configuration and constants for the transcreation pipeline
# =============================================================================
# Edit this file to change models, paths, target culture, or tuning params.
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE_PATH     = "test_image.jpg"
OUTPUT_DIR     = "outputs"
SAM_CHECKPOINT = "sam_vit_h.pth"

# ── Target culture ────────────────────────────────────────────────────────────
# This string is passed directly to the LLM as the transcreation target.
# Change this to adapt the pipeline for any culture / city / context.
TARGET_CULTURE = "Nigerian Lagos street scene"

# ── Device ────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Models ────────────────────────────────────────────────────────────────────
BLIP2_MODEL     = "Salesforce/blip2-opt-2.7b"
OWLVIT_MODEL    = "google/owlvit-base-patch32"
SAM_MODEL_TYPE  = "vit_h"
INPAINT_MODEL   = "runwayml/stable-diffusion-inpainting"
LLM_MODEL       = "gpt-4o"   # swap for "claude-3-5-sonnet-20241022" etc.

# ── OWL-ViT candidate object labels ──────────────────────────────────────────
# These are the object categories the detector will look for in the image.
# Add or remove labels based on your expected image content.
DETECTION_CANDIDATE_LABELS = [
    "car", "bus", "truck", "motorcycle", "bicycle", "taxi",
    "person", "pedestrian", "building", "road", "market stall",
    "tree", "sign", "traffic light", "shop", "clothing", "food stall",
]

# ── SAM mask filtering ────────────────────────────────────────────────────────
SAM_AREA_MIN = 2_000    # ignore tiny noise masks
SAM_AREA_MAX = 50_000   # ignore large background masks
SAM_PRED_IOU_THRESH        = 0.88
SAM_STABILITY_SCORE_THRESH = 0.92

# ── Inpainting parameters ─────────────────────────────────────────────────────
INPAINT_SIZE         = 512    # SD inpainting works best at 512×512
GUIDANCE_SCALE       = 7.5
NUM_INFERENCE_STEPS  = 50
PADDING_MASK_CROP    = 32

# ── API keys ──────────────────────────────────────────────────────────────────
#TODO
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "key goes here")
HF_TOKEN       = os.getenv("HF_TOKEN", "hf token goes here")

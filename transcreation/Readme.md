# Image Transcreation Pipeline

A modular AI pipeline that takes an image and automatically re-contextualises it for a target culture — replacing visually dominant, culturally salient objects (vehicles, signage, clothing, etc.) with culturally appropriate equivalents, using a combination of vision models, an LLM, SAM segmentation, and Stable Diffusion inpainting.

---

## What is Image Transcreation?

Transcreation goes beyond translation. Where translation converts words, transcreation adapts the *feel* of a visual — swapping out objects and elements so the image resonates authentically with a new cultural audience.

For example, a street scene shot in London might have a red double-decker bus, western clothing, and English signage. Transcreated for Lagos, those elements become a yellow danfo minibus, Ankara fabrics, and Yoruba market stalls — while the composition, lighting, and people remain untouched.

---

## Pipeline Overview

```
Input Image
    │
    ├─ Step 1 ── tagger.py      Detect objects + measure their visual area
    │
    ├─ Step 2 ── llm_brain.py   LLM picks culturally salient objects + writes inpaint prompts
    │
    ├─ Step 3 ── segmenter.py   SAM generates a binary mask for each selected object
    │
    └─ Step 4 ── inpainter.py   Iterative SD inpainting produces the final image
```

Each step is isolated in its own module. Swapping a model (e.g. replacing OWL-ViT with RAM, or GPT-4o with Claude) only requires touching one file.

---

## File Structure

```
transcreation/
├── main.py          Entry point — orchestrates all four steps
├── config.py        All settings: paths, models, thresholds, API keys
├── tagger.py        Step 1: Object detection + area measurement
├── llm_brain.py     Step 2: LLM cultural selection + prompt generation
├── segmenter.py     Step 3: SAM mask generation + assignment
├── inpainter.py     Step 4: Iterative Stable Diffusion inpainting
└── utils.py         Shared helpers: visualisation, mask saving, logging
```

---

## Module Reference

### `config.py` — Configuration

The single source of truth for every tunable parameter in the pipeline. Nothing is hardcoded in the logic files — they all import from here.

**Key settings:**

| Setting | Default | Description |
|---|---|---|
| `IMAGE_PATH` | `"test_image.jpg"` | Path to the input image |
| `OUTPUT_DIR` | `"outputs"` | Directory for all saved outputs |
| `TARGET_CULTURE` | `"Nigerian Lagos street scene"` | The transcreation target passed to the LLM |
| `MIN_AREA_PCT` | `2.0` | Objects smaller than this % of the image are dropped before the LLM sees them |
| `BLIP2_MODEL` | `Salesforce/blip2-opt-2.7b` | Caption model |
| `OWLVIT_MODEL` | `google/owlvit-base-patch32` | Open-vocabulary object detector |
| `SAM_MODEL_TYPE` | `vit_h` | SAM variant |
| `INPAINT_MODEL` | `runwayml/stable-diffusion-inpainting` | SD inpainting model |
| `LLM_MODEL` | `gpt-4o` | LLM used for cultural reasoning |
| `GUIDANCE_SCALE` | `7.5` | SD classifier-free guidance strength |
| `NUM_INFERENCE_STEPS` | `50` | SD denoising steps (higher = better quality, slower) |
| `PADDING_MASK_CROP` | `32` | Pixels of context padding added around each mask during inpainting |

To retarget the pipeline for a different culture, only `TARGET_CULTURE` needs to change.

---

### `tagger.py` — Step 1: Tag Extraction

Detects all objects in the image and measures how much of the image each one occupies. This area information is critical — it tells the LLM which objects are visually dominant and worth replacing.

**Models used:**
- **BLIP-2** generates a natural-language caption of the full scene, giving the LLM contextual understanding of what is happening in the image.
- **OWL-ViT** performs open-vocabulary object detection against the candidate label list defined in `config.DETECTION_CANDIDATE_LABELS`, returning bounding boxes for each detected instance.

**Area calculation:** Each detected bounding box's pixel area is computed as a percentage of the total image area. When the same label appears multiple times (e.g. three parked cars), their areas are **summed** — so the LLM sees the total visual footprint of that object class.

Objects below `config.MIN_AREA_PCT` (default 2%) are dropped entirely before anything is passed downstream.

**Returns:**
```python
tags = [
    {"label": "bus",    "area_pct": 34.20},
    {"label": "person", "area_pct":  8.70},
    {"label": "car",    "area_pct":  3.10},
]
caption = "a busy street with buses and pedestrians"
```

Tags are sorted largest-first so the LLM sees the most important objects at the top.

**Public function:**
```python
tags, caption = extract_tags(image)
```

---

### `llm_brain.py` — Step 2: Cultural Selection & Prompt Writing

The reasoning core of the pipeline. The LLM receives the detected objects (with their area percentages) and the target culture, and makes two decisions:

1. **Which objects are culturally salient?** Not every object needs replacing. A road is culturally neutral. A bus or a market stall is not. The LLM filters for objects whose replacement will meaningfully shift the cultural feel of the image.

2. **What should replace them?** For each selected object, the LLM writes a detailed, photorealistic Stable Diffusion inpainting prompt that describes the culturally equivalent replacement in context.

**Area-awareness:** The tags are presented to the LLM as a formatted table:
```
  label                area_pct
  -------------------- ----------
  bus                     34.20%
  person                   8.70%
  car                      3.10%
```
The system prompt instructs the LLM to prioritise visually dominant objects and the user prompt sets a soft minimum of `area_pct > 5.0`. This prevents the LLM from wasting inpainting calls on objects that are too small to be noticed in the final result.

**LLM output** is parsed from JSON into a structured list:
```python
[
    {
        "label":    "bus",
        "area_pct": 34.2,
        "prompt":   "a yellow Nigerian danfo minibus with black stripes, crowded Lagos street, passengers boarding, conductor at doorway, realistic documentary photography, natural lighting, 8k"
    }
]
```

**Swapping the LLM:** The `OpenAI` client call is contained in a single function. To use Anthropic Claude or Google Gemini instead, replace the `client.chat.completions.create(...)` block — the prompts and JSON parsing are model-agnostic.

**Public function:**
```python
label_prompt_pairs = select_and_generate_prompts(tags, caption)
```

---

### `segmenter.py` — Step 3: SAM Segmentation

Generates a precise binary mask for each object selected by the LLM. The mask defines exactly which pixels will be handed to the inpainting model.

**How it works:**

1. **SAM automatic mask generation** runs over the full image, producing all plausible segmentation masks (often 100–300 masks for a typical street scene).

2. **Area filtering** removes masks that are too small (noise) or too large (background), using `SAM_AREA_MIN` and `SAM_AREA_MAX` from config.

3. **Mask assignment** maps one mask to each LLM-selected label. Masks are ranked by SAM's predicted IoU confidence score, and the highest-scoring unused mask is assigned to each label in turn.

Each accepted mask is saved as a grayscale PNG to the output directory (white = region to inpaint, black = keep).

> **Known limitation:** The current assignment strategy (rank by IoU, assign in order) is a reasonable heuristic but not semantically precise. For production use, replace with **CLIP-based matching**: crop each mask region from the image, embed both the crop and the label string with CLIP, and assign the mask with the highest cosine similarity. Alternatively, use SAM3's text-prompted mode (`facebook/sam3`) which accepts a text query and returns masks directly.

**Returns:**
```python
[
    {
        "label":     "bus",
        "prompt":    "a yellow Nigerian danfo minibus ...",
        "mask":      <PIL.Image>,       # grayscale mask image
        "mask_path": "outputs/mask_bus.png"
    }
]
```

**Public function:**
```python
segmented_items = generate_masks(image, label_prompt_pairs, output_dir)
```

---

### `inpainter.py` — Step 4: Iterative Inpainting

Applies Stable Diffusion inpainting sequentially over each `(mask, prompt)` pair. The key design choice here is **iterative composition**: the output of each inpainting step becomes the input to the next. This means all edits accumulate into a single coherent image rather than being applied independently to the original.

```
original image
    └─ inpaint "bus"   → intermediate_1
         └─ inpaint "person" → intermediate_2
              └─ inpaint "car" → final_transcreated.png
```

Intermediate results are saved after each step so you can inspect the progression, catch failures early, or resume from a specific step if something goes wrong.

**Inpainting parameters** (all in `config.py`):

- `GUIDANCE_SCALE` — how strongly the model follows the prompt. Higher values (8–12) give more literal prompt adherence; lower values (5–7) allow more natural blending with surrounding context.
- `NUM_INFERENCE_STEPS` — more steps = higher quality but slower. 30 is fine for prototyping; 50–75 for production.
- `PADDING_MASK_CROP` — pixels of surrounding context fed to the model. More padding helps the model blend the inpainted region with its surroundings; too much can cause the model to "leak" outside the mask.

**Public function:**
```python
final_image = run_iterative_inpainting(image, segmented_items, output_dir)
```

---

### `utils.py` — Shared Utilities

Visualisation and helper functions used across the pipeline.

| Function | Description |
|---|---|
| `overlay_masks(image, masks)` | Renders all masks as semi-transparent coloured overlays on the image. Useful for visually verifying that SAM segmented the right regions before running inpainting. |
| `show_before_after(original, result)` | Side-by-side matplotlib comparison of the input and output image. Optionally saves to disk. |
| `save_mask(mask_np, label, output_dir)` | Converts a numpy boolean mask to a PNG and saves it. Returns the file path. |
| `print_step(n, title)` | Prints a formatted step header to stdout for pipeline progress tracking. |

---

### `main.py` — Entry Point

Wires all four steps together and handles the top-level flow: loading the image, calling each module in sequence, saving outputs, and displaying the final comparison. Reading `main.py` alone should give a clear picture of the entire pipeline without needing to look at any other file.

```python
# Run with defaults from config.py
python main.py

# Or call programmatically with a custom image path
from main import run_pipeline
final_image = run_pipeline("my_image.jpg")
```

**Outputs written to `outputs/`:**

| File | Description |
|---|---|
| `mask_<label>.png` | Binary mask for each segmented object |
| `mask_overlay.png` | All masks visualised on the original image |
| `step_01_<label>.png` | Intermediate image after each inpainting step |
| `final_transcreated.png` | The finished transcreated image |
| `before_after.png` | Side-by-side comparison figure |

---

## Setup

### Requirements

```bash
pip install diffusers transformers accelerate safetensors
pip install opencv-python pillow matplotlib openai
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### SAM checkpoint

```bash
wget -O sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### API keys

Set your OpenAI (or alternative LLM) API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."         # only needed for gated HuggingFace models
```

Or edit them directly in `config.py` for quick prototyping.

---

## Quick Start

```python
# 1. Set your image and target culture in config.py
IMAGE_PATH     = "my_street_photo.jpg"
TARGET_CULTURE = "Tokyo street scene"

# 2. Run
python main.py
```

---

## Extending the Pipeline

| Goal | What to change |
|---|---|
| Different target culture | `config.TARGET_CULTURE` |
| Different LLM (e.g. Claude, Gemini) | `llm_brain.py` — replace the OpenAI client call |
| Different object detector (e.g. RAM, Tag2Text) | `tagger.py` — replace `_detect_objects_with_area()` |
| Semantic mask assignment (CLIP-based) | `segmenter.py` — replace `_assign_masks_to_labels()` |
| Higher quality inpainting (e.g. SDXL) | `config.INPAINT_MODEL` + update `inpainter.py` if the API differs |
| Add new object categories to detect | `config.DETECTION_CANDIDATE_LABELS` |
| Tighten/loosen the area significance filter | `config.MIN_AREA_PCT` |

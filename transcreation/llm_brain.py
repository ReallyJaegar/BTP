# =============================================================================
# llm_brain.py — Step 2: Select culturally salient objects + write prompts
# =============================================================================
# The LLM acts as the "cultural brain" of the pipeline. Given:
#   • The image caption (scene context)
#   • The list of detected object tags
#   • The target culture string
#
# It decides WHICH objects are worth replacing to make the image feel
# authentically from that culture, and writes a detailed inpainting prompt
# for each one.
#
# Returns a list of dicts: [{"label": str, "prompt": str}, ...]
# =============================================================================

import json
from openai import OpenAI   # swap for anthropic SDK if preferred

import config


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in visual cultural adaptation for advertising and media.

Given a list of objects detected in an image and a target culture, your job is:
1. Identify which objects are "culturally salient" — objects whose visual
   replacement with a culturally equivalent version would make the image feel
   authentically from the target culture. Ignore background elements, abstract
   objects, or things that are culturally neutral.
2. For each selected object, write a detailed, photorealistic Stable Diffusion
   inpainting prompt describing the culturally appropriate replacement in context.

Respond ONLY with a valid JSON array. No preamble, no explanation, no markdown fences.
"""

USER_PROMPT_TEMPLATE = """\
Image caption: "{caption}"
Detected objects: {tags}
Target culture: "{culture}"

Return a JSON array. Each element must have exactly two keys:
  "label"  — must exactly match one of the detected object strings above
  "prompt" — a detailed inpainting prompt for the cultural replacement

Example:
[
  {{
    "label": "bus",
    "prompt": "a yellow Nigerian danfo minibus with black stripes, crowded Lagos street, passengers boarding, conductor at the doorway, realistic documentary photography, sharp focus, natural lighting, 8k"
  }}
]
"""


# ── Public API ────────────────────────────────────────────────────────────────

def select_and_generate_prompts(
    tags:    list[str],
    caption: str,
    target_culture: str | None = None,
) -> list[dict]:
    """
    Call the LLM to select culturally salient objects and generate
    an inpainting prompt for each one.

    Args:
        tags:           List of detected object labels from the tagger.
        caption:        Scene caption from BLIP-2.
        target_culture: Override for config.TARGET_CULTURE if provided.

    Returns:
        List of dicts with keys "label" and "prompt".
    """
    culture = target_culture or config.TARGET_CULTURE

    user_message = USER_PROMPT_TEMPLATE.format(
        caption=caption,
        tags=json.dumps(tags),
        culture=culture,
    )

    client   = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.7,
    )

    raw_content = response.choices[0].message.content.strip()
    pairs = _parse_llm_response(raw_content)

    print(f"  LLM selected {len(pairs)} object(s): {[p['label'] for p in pairs]}")
    return pairs


# ── Private helpers ───────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> list[dict]:
    """
    Safely parse the LLM's JSON response, stripping markdown fences if present.
    Raises ValueError with a helpful message if parsing fails.
    """
    # Strip ```json ... ``` fences if the LLM added them despite instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw   = "\n".join(lines[1:-1])  # drop first and last fence lines

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON.\nRaw response:\n{raw}\nError: {e}"
        )

    if not isinstance(result, list):
        raise ValueError(f"Expected a JSON array, got: {type(result)}")

    for item in result:
        if "label" not in item or "prompt" not in item:
            raise ValueError(f"Each item must have 'label' and 'prompt'. Got: {item}")

    return result

"""Built-in caption instruction presets.

Each preset targets a dataset use-case; users can override with a custom prompt.
"""
from __future__ import annotations

PROMPT_PRESETS: dict[str, str] = {
    "Detailed Description":
        "Describe this image in detail, focusing on visual elements, colors, lighting, and composition.",
    "Training Caption (natural)":
        "Write a single-paragraph caption of this image for training an image generation model. "
        "Describe the subject, their appearance, clothing, pose and expression, the setting, lighting, "
        "camera framing and art style. Be factual and specific. Do not mention the image itself "
        "(no 'this image shows'), and do not speculate.",
    "Concise Caption":
        "Write one concise sentence describing the main subject and setting of this image.",
    "Booru Tags":
        "List booru-style tags for this image as a single comma-separated line, lowercase, "
        "most important first (subject count, character features, clothing, pose, background, style). "
        "Output only the tags.",
    "Stable Diffusion Tags":
        "Describe this image using comma-separated tags (e.g., 1girl, solo, sunset, detailed background). "
        "Output only the tags.",
    "Character Description":
        "Describe the character's physical appearance in detail: face, hair, eyes, body type, "
        "skin, distinguishing features, clothing and accessories. Ignore the background.",
    "Clothing & Outfit":
        "Describe only the clothing, accessories and footwear in this image, with colors, materials and styles.",
    "Environment & Background":
        "Describe only the setting and background of this image: location, objects, time of day, weather, lighting.",
    "Art Style & Medium":
        "Describe the artistic style, medium, technique, color palette and overall aesthetic of this image.",
    "Composition & Camera":
        "Describe the composition and camera work of this image: shot type (close-up, full body, etc.), "
        "camera angle, perspective, focal length feel, depth of field and framing.",
    "Accessibility Caption":
        "Provide a brief, literal description of the main subject for accessibility purposes.",
    "Short Summary":
        "Summarize the image in one short sentence.",
}

DEFAULT_PRESET = "Detailed Description"


def build_prompt(base: str, *, subject: str = "", start_with_subject: bool = False, tags: str = "") -> str:
    """Compose the final instruction for one image.

    ``subject`` is a trigger word / name the caption should use instead of a
    generic description. ``tags`` are reference tags (e.g. from a WD tagger) that
    ground the model — the "tag → caption" recipe; the model is told they may be wrong.
    """
    parts = [base.strip()]
    subject = subject.strip()
    if subject:
        parts.append(f'The main subject is "{subject}". Refer to them as "{subject}" instead of a generic '
                     f'description such as "a woman", "a man" or "a person".')
        if start_with_subject:
            parts.append(f'Begin the caption with "{subject}".')
    tags = tags.strip()
    if tags:
        parts.append("Reference tags for this image from an automatic tagger (they may contain mistakes; use only "
                     f"what you can actually see): {tags}")
    return "\n\n".join(p for p in parts if p)

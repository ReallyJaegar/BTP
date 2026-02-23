# =============================================================================
# utils.py — Visualisation helpers and shared utilities
# =============================================================================

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


def overlay_masks(image: Image.Image, masks: list[Image.Image]) -> Image.Image:
    """
    Overlay coloured semi-transparent masks on top of the image.
    Useful for visually inspecting which regions SAM has segmented.

    Args:
        image:  The original PIL image.
        masks:  List of grayscale PIL mask images (white = masked region).

    Returns:
        A new RGBA PIL image with all masks blended in.
    """
    image = image.convert("RGBA")
    cmap  = matplotlib.colormaps.get_cmap("rainbow").resampled(len(masks))
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(len(masks))
    ]

    for mask_pil, color in zip(masks, colors):
        mask_np = np.array(mask_pil.convert("L"))
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha   = Image.fromarray(mask_np).point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image   = Image.alpha_composite(image, overlay)

    return image


def show_before_after(
    original: Image.Image,
    result:   Image.Image,
    save_path: str | None = None,
) -> None:
    """
    Display a side-by-side comparison of the original and transcreated image.
    Optionally saves the figure to disk.

    Args:
        original:  The original input image.
        result:    The final transcreated image.
        save_path: If provided, saves the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(original); axes[0].set_title("Original",     fontsize=14); axes[0].axis("off")
    axes[1].imshow(result);   axes[1].set_title("Transcreated", fontsize=14); axes[1].axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Comparison saved → {save_path}")

    plt.show()


def save_mask(mask_np: np.ndarray, label: str, output_dir: str) -> str:
    """
    Convert a boolean/float numpy mask to a PNG and save it.

    Args:
        mask_np:    2D numpy array (bool or 0-1 float).
        label:      Object label used to name the file.
        output_dir: Directory to save into.

    Returns:
        The full path to the saved mask file.
    """
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    mask_pil   = Image.fromarray(mask_uint8)
    path       = os.path.join(output_dir, f"mask_{label.replace(' ', '_')}.png")
    mask_pil.save(path)
    return path


def print_step(step_number: int, title: str) -> None:
    """Print a clearly visible step header to stdout."""
    print(f"\n{'─' * 60}")
    print(f"  STEP {step_number} — {title.upper()}")
    print(f"{'─' * 60}")

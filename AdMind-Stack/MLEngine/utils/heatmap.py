# utils/heatmap.py

import matplotlib
matplotlib.use("Agg")  # ✅ Run safely on macOS / headless servers (no GUI windows)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_saliency_heatmap(
    image_path: str,
    save_path: str = "sample_heatmap.png",
    alpha: float = 0.45
):
    """
    Generates a transparent saliency heatmap overlay on top of the original image.

    The heatmap highlights visually significant areas (edges, faces, contrasts)
    using OpenCV's saliency detector. If unavailable, it falls back to
    a gradient-based intensity map.

    Args:
        image_path (str): Path to the input image.
        save_path (str): Path to save the final blended heatmap.
        alpha (float): Heatmap transparency (0 = no overlay, 1 = full overlay).

    Returns:
        str: Path to the saved blended heatmap image.
    """

    # ---------------------------------------------------------------------
    # Load & preprocess image
    # ---------------------------------------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # ---------------------------------------------------------------------
    # Compute saliency map
    # ---------------------------------------------------------------------
    try:
        # OpenCV saliency model (if available)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, saliency_map = saliency.computeSaliency(img_resized)
        if not success:
            raise RuntimeError("Saliency computation failed.")
        saliency_map = (saliency_map * 255).astype(np.uint8)
    except Exception:
        # Fallback: compute intensity via edge gradients
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        saliency_map = cv2.magnitude(grad_x, grad_y)
        saliency_map = cv2.normalize(
            saliency_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    # ---------------------------------------------------------------------
    # Generate colored heatmap and overlay
    # ---------------------------------------------------------------------
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_INFERNO)
    blended = cv2.addWeighted(img_resized, 1 - alpha, heatmap, alpha, 0)

    # ---------------------------------------------------------------------
    # Save blended image safely via Matplotlib (no GUI)
    # ---------------------------------------------------------------------
    plt.figure(figsize=(4, 4))
    plt.imshow(blended)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_path

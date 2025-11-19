"""
Color quantization module.
Reduces the number of colors in an image to a discrete palette using k-means clustering.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def quantize_colors(image: np.ndarray,
                   n_colors: int = 16,
                   method: str = 'kmeans',
                   max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize image colors to a reduced palette using k-means clustering.

    This is essential for images with many colors (anti-aliasing, gradients, compression artifacts)
    to make them suitable for the multi-layer filtering algorithm.

    Args:
        image: Input RGB image with shape (H, W, 3)
        n_colors: Number of colors in the quantized palette
        method: Quantization method ('kmeans' or 'median_cut')
        max_iterations: Maximum iterations for k-means

    Returns:
        Tuple of:
        - Quantized RGB image with only n_colors distinct colors
        - Array of palette colors with shape (n_colors, 3)
    """
    if method != 'kmeans':
        raise NotImplementedError(f"Method '{method}' not implemented. Use 'kmeans'.")

    height, width = image.shape[:2]

    # Reshape image to list of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 0.1)
    _, labels, palette = cv2.kmeans(
        pixels,
        n_colors,
        None,
        criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS
    )

    # Convert palette back to uint8
    palette = palette.astype(np.uint8)

    # Reconstruct quantized image
    quantized_pixels = palette[labels.flatten()]
    quantized_image = quantized_pixels.reshape(height, width, 3)

    return quantized_image, palette


def should_quantize(image: np.ndarray, threshold: int = 50) -> bool:
    """
    Check if an image should be quantized based on number of unique colors.

    Images with many colors (>threshold) are likely not discrete-color maps
    and will benefit from quantization.

    Args:
        image: Input RGB image
        threshold: Number of colors above which quantization is recommended

    Returns:
        True if quantization is recommended, False otherwise
    """
    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    return len(unique_colors) > threshold


def estimate_optimal_colors(image: np.ndarray,
                           min_colors: int = 8,
                           max_colors: int = 32,
                           variance_threshold: float = 0.95) -> int:
    """
    Estimate optimal number of colors for quantization based on color distribution.

    Uses the elbow method on k-means variance to find a good balance.

    Args:
        image: Input RGB image
        min_colors: Minimum number of colors to consider
        max_colors: Maximum number of colors to consider
        variance_threshold: Explained variance threshold (0-1)

    Returns:
        Estimated optimal number of colors
    """
    # Sample pixels for efficiency (use at most 10000 pixels)
    pixels = image.reshape(-1, 3).astype(np.float32)

    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]

    # Try different k values and measure inertia
    inertias = []
    k_range = range(min_colors, min(max_colors + 1, len(pixels)))

    for k in k_range:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, _, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            attempts=1,
            flags=cv2.KMEANS_PP_CENTERS
        )

        # Calculate inertia (within-cluster sum of squares)
        labels = np.argmin(np.linalg.norm(pixels[:, None] - centers, axis=2), axis=1)
        inertia = np.sum((pixels - centers[labels]) ** 2)
        inertias.append(inertia)

    # Find elbow point (where adding more colors gives diminishing returns)
    if len(inertias) < 2:
        return min_colors

    # Normalize inertias
    inertias = np.array(inertias)
    inertias = inertias / inertias[0]

    # Find point where variance explained exceeds threshold
    for i, inertia_ratio in enumerate(inertias):
        if (1 - inertia_ratio) >= variance_threshold:
            return min_colors + i

    # Default to middle of range if no clear elbow
    return (min_colors + max_colors) // 2


def adaptive_quantize(image: np.ndarray,
                     max_colors: int = 32,
                     auto_threshold: int = 50) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Adaptively quantize an image only if needed.

    Automatically detects if quantization is necessary and applies it with
    optimal parameters.

    Args:
        image: Input RGB image
        max_colors: Maximum number of colors for quantization
        auto_threshold: Threshold for automatic quantization decision

    Returns:
        Tuple of:
        - Quantized image (or original if quantization not needed)
        - Palette colors if quantized, None otherwise
    """
    if not should_quantize(image, auto_threshold):
        # Image already has few colors, no quantization needed
        return image, None

    # Estimate optimal number of colors
    n_colors = estimate_optimal_colors(image, min_colors=8, max_colors=max_colors)

    # Apply quantization
    quantized_image, palette = quantize_colors(image, n_colors=n_colors)

    return quantized_image, palette

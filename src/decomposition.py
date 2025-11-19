"""
Color-to-layer decomposition module.
Extracts unique colors from input image and generates binary layers.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def get_unique_colors(image: np.ndarray) -> np.ndarray:
    """
    Extract unique colors from an RGB image.

    Args:
        image: Input RGB image as numpy array with shape (H, W, 3)

    Returns:
        Array of unique colors with shape (N, 3) where N is number of unique colors
    """
    # Reshape image to list of pixels
    pixels = image.reshape(-1, 3)

    # Find unique colors
    unique_colors = np.unique(pixels, axis=0)

    return unique_colors


def decompose_to_layers(image: np.ndarray, colors: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Decompose a color image into N binary layers, one for each unique color.

    For each color t, create a binary layer L_t where:
    L_t(i,j) = 1 if pixel(i,j) matches color t, otherwise 0

    Args:
        image: Input RGB image as numpy array with shape (H, W, 3)
        colors: Optional array of colors to decompose. If None, extracts unique colors.

    Returns:
        Tuple of:
        - List of binary layers (numpy arrays with dtype uint8, values 0 or 255)
        - Array of colors corresponding to each layer
    """
    if colors is None:
        colors = get_unique_colors(image)

    height, width = image.shape[:2]
    layers = []

    # Create a binary layer for each unique color
    for color in colors:
        # Create mask where all channels match the target color
        mask = np.all(image == color, axis=-1)

        # Convert boolean mask to binary image (0 or 255)
        binary_layer = (mask * 255).astype(np.uint8)

        layers.append(binary_layer)

    return layers, colors


def layers_to_color_indices(image: np.ndarray, colors: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert color image to index map where each pixel stores the index of its color.

    This is useful for efficient reconstruction and processing.

    Args:
        image: Input RGB image as numpy array with shape (H, W, 3)
        colors: Optional array of colors. If None, extracts unique colors.

    Returns:
        Index map with shape (H, W) where each value is the index into colors array
    """
    if colors is None:
        colors = get_unique_colors(image)

    height, width = image.shape[:2]
    index_map = np.zeros((height, width), dtype=np.int32)

    # Map each pixel to its color index
    for idx, color in enumerate(colors):
        mask = np.all(image == color, axis=-1)
        index_map[mask] = idx

    return index_map


def count_color_pixels(layers: List[np.ndarray]) -> np.ndarray:
    """
    Count the number of pixels for each color layer.

    Args:
        layers: List of binary layers

    Returns:
        Array of pixel counts for each layer
    """
    counts = np.array([np.sum(layer > 0) for layer in layers])
    return counts

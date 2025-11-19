"""
Image reconstruction module.
Merges filtered binary layers back into a color image using priority ordering.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional


def reconstruct_image(filtered_layers: List[np.ndarray],
                     colors: np.ndarray,
                     label_mask: Optional[np.ndarray] = None,
                     priority_map: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruct color image from filtered binary layers.

    For each pixel, select the color from the highest-priority non-zero layer
    according to equation (3) in the paper.

    Args:
        filtered_layers: List of filtered binary layers
        colors: Array of colors corresponding to each layer (shape: N x 3)
        label_mask: Optional label mask from segmentation
        priority_map: Optional priority values for each pixel

    Returns:
        Reconstructed RGB image
    """
    if len(filtered_layers) == 0 or len(colors) == 0:
        raise ValueError("No layers or colors provided for reconstruction")

    height, width = filtered_layers[0].shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    if priority_map is None:
        # Simple reconstruction: iterate through layers in order
        # Last layer wins (can be modified to use frequency-based ordering)
        for layer_idx, layer in enumerate(filtered_layers):
            mask = layer > 0
            output_image[mask] = colors[layer_idx]
    else:
        # Priority-based reconstruction
        # For each pixel, select color from highest priority layer

        # Create a stack to track which layer each pixel should use
        best_layer = np.full((height, width), -1, dtype=np.int32)
        max_priority = np.full((height, width), -1.0, dtype=np.float32)

        for layer_idx, layer in enumerate(filtered_layers):
            layer_mask = layer > 0

            # For pixels in this layer, check if priority is higher
            if label_mask is not None:
                # Use region-based priorities
                improved = layer_mask & (priority_map > max_priority)
            else:
                # Use layer-level priorities (can be based on frequency)
                layer_priority = 1.0 / (layer_idx + 1)  # Simple: earlier layers have higher priority
                improved = layer_mask & (layer_priority > max_priority)
                priority_map_local = np.where(improved, layer_priority, max_priority)
                max_priority = priority_map_local

            if label_mask is not None:
                improved = layer_mask & (priority_map > max_priority)
                max_priority = np.where(improved, priority_map, max_priority)

            best_layer[improved] = layer_idx

        # Assign colors based on best layer for each pixel
        for layer_idx in range(len(filtered_layers)):
            mask = best_layer == layer_idx
            output_image[mask] = colors[layer_idx]

    return output_image


def reconstruct_with_regions(filtered_layers: List[np.ndarray],
                            colors: np.ndarray,
                            label_mask: np.ndarray,
                            region_info: List[Dict]) -> np.ndarray:
    """
    Reconstruct image using region-based color assignment.

    Each region is filled with its assigned color based on the region_info.

    Args:
        filtered_layers: List of filtered binary layers
        colors: Array of colors corresponding to each layer
        label_mask: Label mask from segmentation
        region_info: List of region information dictionaries

    Returns:
        Reconstructed RGB image
    """
    height, width = label_mask.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill each region with its assigned color
    for region in region_info:
        label = region['label']
        color = region['color']

        region_mask = label_mask == label
        output_image[region_mask] = color

    # Handle any unlabeled pixels (should be rare after fill_unlabeled_pixels)
    unlabeled = label_mask == 0
    if np.any(unlabeled):
        # Use simple layer-based reconstruction for unlabeled pixels
        for layer_idx, layer in enumerate(filtered_layers):
            mask = unlabeled & (layer > 0)
            output_image[mask] = colors[layer_idx]

    return output_image


def reconstruct_with_priority_ordering(filtered_layers: List[np.ndarray],
                                       colors: np.ndarray,
                                       color_priorities: Dict[int, float]) -> np.ndarray:
    """
    Reconstruct image using global priority ordering.

    Args:
        filtered_layers: List of filtered binary layers
        colors: Array of colors corresponding to each layer
        color_priorities: Dictionary mapping layer index to priority value

    Returns:
        Reconstructed RGB image
    """
    height, width = filtered_layers[0].shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Sort layers by priority (highest first)
    sorted_indices = sorted(
        range(len(filtered_layers)),
        key=lambda idx: color_priorities.get(idx, 0),
        reverse=True
    )

    # Apply layers in priority order (highest priority last so it overwrites)
    for layer_idx in reversed(sorted_indices):
        layer = filtered_layers[layer_idx]
        mask = layer > 0
        output_image[mask] = colors[layer_idx]

    return output_image


def calculate_layer_priorities(filtered_layers: List[np.ndarray],
                              strategy: str = 'frequency') -> Dict[int, float]:
    """
    Calculate priority values for each layer.

    Args:
        filtered_layers: List of filtered binary layers
        strategy: Priority strategy ('frequency' or 'uniform')

    Returns:
        Dictionary mapping layer index to priority value (higher = more important)
    """
    priorities = {}

    if strategy == 'frequency':
        # Calculate pixel counts for each layer
        counts = [np.sum(layer > 0) for layer in filtered_layers]
        total = sum(counts)

        if total > 0:
            # Lower frequency = higher priority (to preserve details)
            for idx, count in enumerate(counts):
                priorities[idx] = 1.0 - (count / total)
        else:
            # Fallback to uniform
            for idx in range(len(filtered_layers)):
                priorities[idx] = 1.0 / len(filtered_layers)

    elif strategy == 'uniform':
        # Equal priority for all layers
        for idx in range(len(filtered_layers)):
            priorities[idx] = 1.0

    return priorities


def merge_layers_simple(filtered_layers: List[np.ndarray],
                       colors: np.ndarray) -> np.ndarray:
    """
    Simple layer merging without priority ordering.

    Last non-zero layer wins for each pixel.

    Args:
        filtered_layers: List of filtered binary layers
        colors: Array of colors corresponding to each layer

    Returns:
        Reconstructed RGB image
    """
    height, width = filtered_layers[0].shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply layers in order (last wins)
    for layer_idx, layer in enumerate(filtered_layers):
        mask = layer > 0
        output_image[mask] = colors[layer_idx]

    return output_image

"""
Region-based segmentation and color priority ordering module.
Implements the segmentation algorithm from Figure 3 of the paper.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict


def segment_layers(filtered_layers: List[np.ndarray],
                   colors: np.ndarray,
                   f1_threshold: float = 0.9,
                   f2_threshold: float = 0.05,
                   dilation_size: int = 0,
                   skip_dilation: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Segment filtered layers into distinct regions using the algorithm from Figure 3.

    Steps:
    1. Optionally dilate each filtered layer (default: skip dilation)
    2. Fill holes in layers (if not skipping dilation)
    3. Apply connected components to find candidate regions
    4. Calculate acceptance criteria (f1, f2) for each region
    5. Generate label mask S_M

    Args:
        filtered_layers: List of filtered binary layers
        colors: Array of colors corresponding to each layer
        f1_threshold: Threshold for object pixel ratio (default 0.9 - regions must be 90% accurate)
        f2_threshold: Threshold for labeled pixel percentage (default 0.05 - accept even small fragments)
        dilation_size: Size of dilation kernel for region expansion (default 0 - no dilation)
        skip_dilation: Whether to skip dilation and hole-filling steps (default True - recommended)

    Returns:
        Tuple of:
        - Label mask S_M with shape (H, W) where each pixel has a region label
        - List of region info dictionaries containing metadata for each region
    """
    if len(filtered_layers) == 0:
        raise ValueError("No layers provided for segmentation")

    height, width = filtered_layers[0].shape
    label_mask = np.zeros((height, width), dtype=np.int32)
    region_info = []
    current_label = 1

    # Create dilation kernel if needed
    if not skip_dilation and dilation_size > 0:
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_size, dilation_size)
        )

    # Process each layer
    for layer_idx, layer in enumerate(filtered_layers):
        # Step 1: Optionally dilate the filtered layer
        if skip_dilation or dilation_size == 0:
            # Use filtered layer directly - this preserves accurate boundaries
            processed_layer = layer.copy()
        else:
            processed_layer = cv2.dilate(layer, dilation_kernel, iterations=1)

        # Step 2: Fill holes (only if we dilated)
        if not skip_dilation and dilation_size > 0:
            # Use floodFill from the border to find background, then invert
            filled = processed_layer.copy()
            h, w = filled.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(filled, mask, (0, 0), 255)
            filled_inv = cv2.bitwise_not(filled)
            processed_layer = cv2.bitwise_or(processed_layer, filled_inv)

        # Step 3: Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            processed_layer, connectivity=8
        )

        # Step 4: Evaluate each candidate region (skip background label 0)
        for region_label in range(1, num_labels):
            region_mask = (labels == region_label).astype(np.uint8)

            # Calculate f1: ratio of object pixels to total region pixels
            object_pixels = np.sum((region_mask > 0) & (layer > 0))
            total_region_pixels = np.sum(region_mask > 0)

            if total_region_pixels == 0:
                continue

            f1 = object_pixels / total_region_pixels

            # Calculate f2: percentage of object pixels that are labeled
            total_object_pixels = np.sum(layer > 0)
            if total_object_pixels == 0:
                f2 = 0
            else:
                labeled_object_pixels = object_pixels
                f2 = labeled_object_pixels / total_object_pixels

            # Apply acceptance criteria
            if f1 >= f1_threshold and f2 >= f2_threshold:
                # Accept this region
                region_mask_bool = region_mask > 0

                # Update label mask (avoid overwriting existing labels)
                label_mask[region_mask_bool & (label_mask == 0)] = current_label

                # Store region information
                region_info.append({
                    'label': current_label,
                    'layer_idx': layer_idx,
                    'color': colors[layer_idx],
                    'f1': f1,
                    'f2': f2,
                    'pixel_count': np.sum(region_mask_bool & (label_mask == current_label)),
                    'centroid': centroids[region_label]
                })

                current_label += 1

    return label_mask, region_info


def calculate_color_priorities(label_mask: np.ndarray,
                               region_info: List[Dict],
                               filtered_layers: List[np.ndarray],
                               colors: np.ndarray,
                               strategy: str = 'frequency') -> np.ndarray:
    """
    Calculate color priorities for each region.

    Strategy options:
    - 'frequency': Lower frequency colors get higher priority (to preserve details)
    - 'global': Global frequency-based ordering for all regions

    Args:
        label_mask: Label mask from segmentation with shape (H, W)
        region_info: List of region information dictionaries
        filtered_layers: List of filtered binary layers
        colors: Array of colors corresponding to each layer
        strategy: Priority calculation strategy

    Returns:
        Priority map with shape (H, W) where each pixel has a priority value
        Higher values = higher priority
    """
    height, width = label_mask.shape
    priority_map = np.zeros((height, width), dtype=np.float32)

    if strategy == 'frequency':
        # Calculate color frequencies for each region
        for region in region_info:
            region_label = region['label']
            region_mask = (label_mask == region_label)
            layer_idx = region['layer_idx']

            # Count pixels of each color in this region
            color_counts = []
            for idx, layer in enumerate(filtered_layers):
                count = np.sum((layer > 0) & region_mask)
                color_counts.append(count)

            # Assign priorities (inverse of frequency)
            # Lower frequency = higher priority
            total_pixels = np.sum(region_mask)
            if total_pixels > 0:
                # Priority for the main color of this region
                main_color_count = color_counts[layer_idx]
                if main_color_count > 0:
                    # Normalize: less frequent = higher priority
                    priority = 1.0 - (main_color_count / total_pixels)
                    priority_map[region_mask] = priority

    elif strategy == 'global':
        # Global frequency-based ordering
        layer_counts = np.array([np.sum(layer > 0) for layer in filtered_layers])

        # Sort by frequency (ascending)
        sorted_indices = np.argsort(layer_counts)

        # Assign priorities (less frequent = higher priority)
        max_count = np.max(layer_counts) if len(layer_counts) > 0 else 1

        for region in region_info:
            region_label = region['label']
            region_mask = (label_mask == region_label)
            layer_idx = region['layer_idx']

            # Priority based on global frequency
            if max_count > 0:
                priority = 1.0 - (layer_counts[layer_idx] / max_count)
                priority_map[region_mask] = priority

    return priority_map


def fill_unlabeled_pixels(label_mask: np.ndarray,
                          filtered_layers: List[np.ndarray]) -> np.ndarray:
    """
    Fill unlabeled pixels (label=0) using distance transform.

    Each unlabeled pixel is assigned the label of the nearest labeled region.

    Args:
        label_mask: Label mask with some pixels labeled 0
        filtered_layers: List of filtered binary layers (for fallback)

    Returns:
        Updated label mask with all pixels labeled
    """
    # Find unlabeled pixels
    unlabeled_mask = (label_mask == 0).astype(np.uint8)

    if np.sum(unlabeled_mask) == 0:
        return label_mask  # All pixels already labeled

    # Create binary mask of labeled regions
    labeled_mask = (label_mask > 0).astype(np.uint8)

    # Compute distance transform
    dist_transform = cv2.distanceTransform(
        cv2.bitwise_not(labeled_mask * 255),
        cv2.DIST_L2,
        5
    )

    # For each unlabeled pixel, find nearest labeled pixel
    result_mask = label_mask.copy()

    # Use nearest neighbor approach: dilate labeled regions iteratively
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp_mask = label_mask.copy()

    max_iterations = 100
    for _ in range(max_iterations):
        if np.sum(result_mask == 0) == 0:
            break

        # Dilate each label
        unique_labels = np.unique(temp_mask[temp_mask > 0])
        for label in unique_labels:
            label_region = (temp_mask == label).astype(np.uint8)
            dilated = cv2.dilate(label_region, kernel, iterations=1)

            # Assign label to newly covered unlabeled pixels
            new_pixels = (dilated > 0) & (result_mask == 0)
            result_mask[new_pixels] = label

        temp_mask = result_mask.copy()

    return result_mask


def create_region_color_map(label_mask: np.ndarray,
                           region_info: List[Dict],
                           colors: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Create a mapping from region labels to their assigned colors.

    Args:
        label_mask: Label mask with region assignments
        region_info: List of region information dictionaries
        colors: Array of colors

    Returns:
        Dictionary mapping region label to RGB color
    """
    color_map = {}

    for region in region_info:
        label = region['label']
        color = region['color']
        color_map[label] = color

    return color_map

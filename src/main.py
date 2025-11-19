"""
Main pipeline for multi-layer map filtering.
Orchestrates the complete filtering process.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict

from .decomposition import decompose_to_layers, get_unique_colors
from .filtering import apply_binary_filter, create_default_filter, BinaryFilter
from .segmentation import segment_layers, calculate_color_priorities, fill_unlabeled_pixels
from .reconstruction import (
    reconstruct_image,
    reconstruct_with_regions,
    calculate_layer_priorities,
    reconstruct_with_priority_ordering
)


class MultiLayerFilter:
    """
    Main class for multi-layer map filtering.

    Implements the ICIP 2009 paper's approach:
    1. Decompose color image to binary layers
    2. Filter each layer independently
    3. Segment layers into regions
    4. Calculate color priorities
    5. Reconstruct filtered image
    """

    def __init__(self,
                 filter_type: str = 'morphological',
                 filter_params: Optional[Dict] = None,
                 use_segmentation: bool = True,
                 f1_threshold: float = 0.5,
                 f2_threshold: float = 0.8):
        """
        Initialize the multi-layer filter.

        Args:
            filter_type: Type of binary filter ('morphological', 'median', 'combined')
            filter_params: Parameters for the filter
            use_segmentation: Whether to use region-based segmentation
            f1_threshold: Threshold for object pixel ratio in segmentation
            f2_threshold: Threshold for labeled pixel percentage in segmentation
        """
        self.filter_type = filter_type
        self.filter_params = filter_params or {}
        self.use_segmentation = use_segmentation
        self.f1_threshold = f1_threshold
        self.f2_threshold = f2_threshold

        # Create filter instance
        self.binary_filter = create_default_filter(filter_type, **self.filter_params)

    def filter_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multi-layer filtering to an image.

        Args:
            image: Input RGB image

        Returns:
            Filtered RGB image
        """
        # Step 1: Decompose to binary layers
        layers, colors = decompose_to_layers(image)

        # Step 2: Filter each layer
        filtered_layers = apply_binary_filter(layers, self.binary_filter)

        # Step 3 & 4: Segment and prioritize (if enabled)
        if self.use_segmentation:
            label_mask, region_info = segment_layers(
                filtered_layers,
                colors,
                self.f1_threshold,
                self.f2_threshold
            )

            # Fill unlabeled pixels
            label_mask = fill_unlabeled_pixels(label_mask, filtered_layers)

            # Calculate priorities
            priority_map = calculate_color_priorities(
                label_mask,
                region_info,
                filtered_layers,
                colors,
                strategy='frequency'
            )

            # Step 5: Reconstruct with regions
            output_image = reconstruct_with_regions(
                filtered_layers,
                colors,
                label_mask,
                region_info
            )
        else:
            # Simple reconstruction with global priorities
            priorities = calculate_layer_priorities(filtered_layers, strategy='frequency')
            output_image = reconstruct_with_priority_ordering(
                filtered_layers,
                colors,
                priorities
            )

        return output_image

    def get_layer_info(self, image: np.ndarray) -> Dict:
        """
        Get information about the layers in an image.

        Args:
            image: Input RGB image

        Returns:
            Dictionary with layer information
        """
        colors = get_unique_colors(image)
        layers, _ = decompose_to_layers(image, colors)

        layer_info = {
            'num_layers': len(layers),
            'colors': colors,
            'pixel_counts': [np.sum(layer > 0) for layer in layers]
        }

        return layer_info


def filter_map_image(image: np.ndarray,
                    filter_type: str = 'morphological',
                    operation: str = 'opening',
                    kernel_size: int = 3,
                    use_segmentation: bool = True) -> np.ndarray:
    """
    Convenience function to filter a map image with default settings.

    Args:
        image: Input RGB image
        filter_type: Type of binary filter
        operation: Morphological operation (if using morphological filter)
        kernel_size: Size of filter kernel
        use_segmentation: Whether to use region-based segmentation

    Returns:
        Filtered RGB image
    """
    filter_params = {
        'operation': operation,
        'kernel_size': kernel_size
    }

    mlf = MultiLayerFilter(
        filter_type=filter_type,
        filter_params=filter_params,
        use_segmentation=use_segmentation
    )

    return mlf.filter_image(image)


def load_image(filepath: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        filepath: Path to image file

    Returns:
        RGB image as numpy array
    """
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Could not load image from {filepath}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(image: np.ndarray, filepath: str):
    """
    Save an image to file.

    Args:
        image: RGB image as numpy array
        filepath: Path to save image
    """
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, bgr_image)

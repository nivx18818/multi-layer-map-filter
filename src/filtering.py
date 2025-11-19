"""
Binary image filtering module.
Implements morphological operations and median filtering for binary layers.
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional


class BinaryFilter(ABC):
    """Abstract base class for binary image filters."""

    @abstractmethod
    def apply(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply filter to a binary image.

        Args:
            binary_image: Binary image (values 0 or 255)

        Returns:
            Filtered binary image
        """
        pass


class MorphologicalFilter(BinaryFilter):
    """
    Morphological filter for binary images.
    Supports dilation, erosion, opening, and closing operations.
    """

    def __init__(self, operation: str = 'opening', kernel_size: int = 3, iterations: int = 1):
        """
        Initialize morphological filter.

        Args:
            operation: Type of operation ('dilation', 'erosion', 'opening', 'closing')
            kernel_size: Size of the structuring element (must be odd)
            iterations: Number of times to apply the operation
        """
        self.operation = operation.lower()
        self.kernel_size = kernel_size
        self.iterations = iterations

        # Create structuring element (kernel)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )

        # Validate operation type
        valid_ops = ['dilation', 'erosion', 'opening', 'closing']
        if self.operation not in valid_ops:
            raise ValueError(f"Operation must be one of {valid_ops}")

    def apply(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operation to binary image.

        Args:
            binary_image: Binary image (values 0 or 255)

        Returns:
            Filtered binary image
        """
        if self.operation == 'dilation':
            result = cv2.dilate(binary_image, self.kernel, iterations=self.iterations)
        elif self.operation == 'erosion':
            result = cv2.erode(binary_image, self.kernel, iterations=self.iterations)
        elif self.operation == 'opening':
            # Opening = erosion followed by dilation
            result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, self.kernel, iterations=self.iterations)
        elif self.operation == 'closing':
            # Closing = dilation followed by erosion
            result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, self.kernel, iterations=self.iterations)
        else:
            # Should never reach here due to validation in __init__, but return unchanged for safety
            result = binary_image

        return result
class MedianFilter(BinaryFilter):
    """
    Median filter for binary images.
    """

    def __init__(self, kernel_size: int = 3):
        """
        Initialize median filter.

        Args:
            kernel_size: Size of the median filter kernel (must be odd)
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        self.kernel_size = kernel_size

    def apply(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply median filter to binary image.

        Args:
            binary_image: Binary image (values 0 or 255)

        Returns:
            Filtered binary image
        """
        result = cv2.medianBlur(binary_image, self.kernel_size)
        return result


class CombinedFilter(BinaryFilter):
    """
    Combine multiple filters in sequence.
    """

    def __init__(self, filters: list):
        """
        Initialize combined filter.

        Args:
            filters: List of BinaryFilter instances to apply in sequence
        """
        self.filters = filters

    def apply(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply all filters in sequence.

        Args:
            binary_image: Binary image (values 0 or 255)

        Returns:
            Filtered binary image
        """
        result = binary_image.copy()
        for filter_obj in self.filters:
            result = filter_obj.apply(result)
        return result


def apply_binary_filter(layers: list, filter_obj: BinaryFilter) -> list:
    """
    Apply a binary filter to all layers.

    Args:
        layers: List of binary layers
        filter_obj: BinaryFilter instance to apply

    Returns:
        List of filtered binary layers
    """
    filtered_layers = []
    for layer in layers:
        filtered = filter_obj.apply(layer)
        filtered_layers.append(filtered)

    return filtered_layers


def create_default_filter(filter_type: str = 'morphological', **kwargs) -> BinaryFilter:
    """
    Create a default filter instance.

    Args:
        filter_type: Type of filter ('morphological', 'median', or 'combined')
        **kwargs: Additional arguments for the filter

    Returns:
        BinaryFilter instance
    """
    if filter_type == 'morphological':
        operation = kwargs.get('operation', 'opening')
        kernel_size = kwargs.get('kernel_size', 3)
        iterations = kwargs.get('iterations', 1)
        return MorphologicalFilter(operation, kernel_size, iterations)

    elif filter_type == 'median':
        kernel_size = kwargs.get('kernel_size', 3)
        return MedianFilter(kernel_size)

    elif filter_type == 'combined':
        # Default combined filter: median + morphological opening
        filters = [
            MedianFilter(kernel_size=3),
            MorphologicalFilter(operation='opening', kernel_size=3)
        ]
        return CombinedFilter(filters)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

"""
Multi-layer Map Filter Implementation
Based on ICIP 2009 paper for filtering color map images.
"""

from .decomposition import decompose_to_layers, get_unique_colors
from .filtering import apply_binary_filter, MorphologicalFilter, MedianFilter
from .segmentation import segment_layers, calculate_color_priorities
from .reconstruction import reconstruct_image
from .evaluation import inject_impulsive_noise, inject_content_dependent_noise, calculate_color_distance, evaluate_filter

__version__ = "1.0.0"

__all__ = [
    'decompose_to_layers',
    'get_unique_colors',
    'apply_binary_filter',
    'MorphologicalFilter',
    'MedianFilter',
    'segment_layers',
    'calculate_color_priorities',
    'reconstruct_image',
    'inject_impulsive_noise',
    'inject_content_dependent_noise',
    'calculate_color_distance',
    'evaluate_filter',
]

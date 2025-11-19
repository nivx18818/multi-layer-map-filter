"""
Evaluation and testing framework.
Implements noise injection and quality metrics.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Callable


def inject_impulsive_noise(image: np.ndarray, noise_ratio: float = 0.1) -> np.ndarray:
    """
    Inject impulsive (salt-and-pepper) noise into an image.

    Args:
        image: Input RGB image
        noise_ratio: Ratio of pixels to corrupt (0.0 to 1.0)

    Returns:
        Noisy image
    """
    noisy = image.copy()
    height, width = image.shape[:2]
    num_pixels = int(height * width * noise_ratio)

    # Random positions for noise
    noise_coords = np.random.randint(0, height * width, num_pixels)

    for coord in noise_coords:
        y = coord // width
        x = coord % width

        # Random color for noise
        if np.random.rand() < 0.5:
            # Salt (white)
            noisy[y, x] = [255, 255, 255]
        else:
            # Pepper (black)
            noisy[y, x] = [0, 0, 0]

    return noisy


def inject_content_dependent_noise(image: np.ndarray,
                                   colors: np.ndarray,
                                   noise_ratio: float = 0.1) -> np.ndarray:
    """
    Inject content-dependent noise that uses colors from the image palette.

    This is more realistic for map images where noise typically comes from
    misclassification rather than random values.

    Args:
        image: Input RGB image
        colors: Array of valid colors in the image
        noise_ratio: Ratio of pixels to corrupt (0.0 to 1.0)

    Returns:
        Noisy image
    """
    noisy = image.copy()
    height, width = image.shape[:2]
    num_pixels = int(height * width * noise_ratio)

    # Random positions for noise
    noise_y = np.random.randint(0, height, num_pixels)
    noise_x = np.random.randint(0, width, num_pixels)

    # Random colors from palette
    color_indices = np.random.randint(0, len(colors), num_pixels)

    for i in range(num_pixels):
        y, x = noise_y[i], noise_x[i]
        # Replace with a different color from the palette
        current_color = image[y, x]

        # Find a different color
        new_color_idx = color_indices[i]
        new_color = colors[new_color_idx]

        # Ensure it's different from current color
        attempts = 0
        while np.array_equal(new_color, current_color) and attempts < 10:
            new_color_idx = np.random.randint(0, len(colors))
            new_color = colors[new_color_idx]
            attempts += 1

        noisy[y, x] = new_color

    return noisy


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to CIE L*a*b* color space.

    Args:
        image: RGB image (0-255)

    Returns:
        L*a*b* image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def calculate_color_distance(image1: np.ndarray,
                            image2: np.ndarray,
                            metric: str = 'delta_e') -> float:
    """
    Calculate color distance between two images.

    Args:
        image1: First RGB image
        image2: Second RGB image
        metric: Distance metric ('delta_e' for CIE ΔE, 'mse' for mean squared error)

    Returns:
        Mean color distance
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape")

    if metric == 'delta_e':
        # Convert to L*a*b* color space
        lab1 = rgb_to_lab(image1)
        lab2 = rgb_to_lab(image2)

        # Calculate ΔE (CIE 1976)
        # ΔE = sqrt((L2-L1)^2 + (a2-a1)^2 + (b2-b1)^2)
        delta = lab1.astype(np.float32) - lab2.astype(np.float32)
        delta_e = np.sqrt(np.sum(delta ** 2, axis=2))

        # Return mean ΔE
        mean_delta_e = np.mean(delta_e)
        return mean_delta_e

    elif metric == 'mse':
        # Mean squared error in RGB space
        mse = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
        return float(mse)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def calculate_psnr(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        original: Original image
        filtered: Filtered image

    Returns:
        PSNR value in dB
    """
    mse = np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)

    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Simplified version without the full scikit-image implementation.

    Args:
        image1: First image
        image2: Second image

    Returns:
        SSIM value (0 to 1)
    """
    # Convert to grayscale for SSIM calculation
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = image1

    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = image2

    # Constants for numerical stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Convert to float
    img1 = gray1.astype(np.float32)
    img2 = gray2.astype(np.float32)

    # Means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    return np.mean(ssim_map).item()  # type: ignore[arg-type]
def evaluate_filter(original: np.ndarray,
                   noisy: np.ndarray,
                   filtered: np.ndarray) -> Dict[str, float]:
    """
    Evaluate filter performance using multiple metrics.

    Args:
        original: Original clean image
        noisy: Noisy input image
        filtered: Filtered output image

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    # Color distance (ΔE) metrics
    metrics['delta_e_noisy'] = calculate_color_distance(original, noisy, 'delta_e')
    metrics['delta_e_filtered'] = calculate_color_distance(original, filtered, 'delta_e')
    metrics['delta_e_improvement'] = metrics['delta_e_noisy'] - metrics['delta_e_filtered']

    # PSNR metrics
    metrics['psnr_noisy'] = calculate_psnr(original, noisy)
    metrics['psnr_filtered'] = calculate_psnr(original, filtered)
    metrics['psnr_improvement'] = metrics['psnr_filtered'] - metrics['psnr_noisy']

    # MSE metrics
    metrics['mse_noisy'] = calculate_color_distance(original, noisy, 'mse')
    metrics['mse_filtered'] = calculate_color_distance(original, filtered, 'mse')
    metrics['mse_improvement'] = metrics['mse_noisy'] - metrics['mse_filtered']

    # SSIM metrics
    metrics['ssim_noisy'] = calculate_ssim(original, noisy)
    metrics['ssim_filtered'] = calculate_ssim(original, filtered)
    metrics['ssim_improvement'] = metrics['ssim_filtered'] - metrics['ssim_noisy']

    return metrics


def compare_filters(original: np.ndarray,
                   noisy: np.ndarray,
                   filters: Dict[str, Callable]) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple filter implementations.

    Args:
        original: Original clean image
        noisy: Noisy input image
        filters: Dictionary mapping filter names to filter functions
                 Each function should take noisy image and return filtered image

    Returns:
        Dictionary mapping filter names to their evaluation metrics
    """
    results = {}

    for filter_name, filter_func in filters.items():
        # Apply filter
        filtered = filter_func(noisy)

        # Evaluate
        metrics = evaluate_filter(original, noisy, filtered)
        results[filter_name] = metrics

    return results


def print_evaluation_results(metrics: Dict[str, float], filter_name: str = "Filter"):
    """
    Print evaluation results in a readable format.

    Args:
        metrics: Dictionary of metric names and values
        filter_name: Name of the filter being evaluated
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {filter_name}")
    print(f"{'='*60}")

    print(f"\nColor Distance (ΔE - CIE LAB):")
    print(f"  Noisy:       {metrics['delta_e_noisy']:.2f}")
    print(f"  Filtered:    {metrics['delta_e_filtered']:.2f}")
    print(f"  Improvement: {metrics['delta_e_improvement']:.2f}")

    print(f"\nPSNR (dB):")
    print(f"  Noisy:       {metrics['psnr_noisy']:.2f}")
    print(f"  Filtered:    {metrics['psnr_filtered']:.2f}")
    print(f"  Improvement: {metrics['psnr_improvement']:.2f}")

    print(f"\nMSE:")
    print(f"  Noisy:       {metrics['mse_noisy']:.2f}")
    print(f"  Filtered:    {metrics['mse_filtered']:.2f}")
    print(f"  Improvement: {metrics['mse_improvement']:.2f}")

    print(f"\nSSIM (0-1, higher is better):")
    print(f"  Noisy:       {metrics['ssim_noisy']:.4f}")
    print(f"  Filtered:    {metrics['ssim_filtered']:.4f}")
    print(f"  Improvement: {metrics['ssim_improvement']:.4f}")

    print(f"{'='*60}\n")

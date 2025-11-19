"""
Example usage of the multi-layer map filter.
Demonstrates the complete pipeline with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.main import MultiLayerFilter, load_image, save_image
from src.evaluation import (
    inject_impulsive_noise,
    inject_content_dependent_noise,
    evaluate_filter,
    print_evaluation_results
)
from src.decomposition import get_unique_colors


def create_synthetic_map(size: int = 256) -> np.ndarray:
    """
    Create a synthetic map image for testing.

    Args:
        size: Size of the square image

    Returns:
        Synthetic RGB map image
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)

    # Define color palette (typical map colors)
    colors = {
        'water': np.array([100, 150, 255]),
        'land': np.array([200, 220, 180]),
        'forest': np.array([50, 150, 50]),
        'city': np.array([200, 100, 100]),
        'road': np.array([100, 100, 100])
    }

    # Fill background with land
    image[:, :] = colors['land']

    # Add water bodies
    image[20:80, 20:120] = colors['water']
    image[150:200, 100:220] = colors['water']

    # Add forest
    image[100:180, 30:90] = colors['forest']

    # Add cities (small squares)
    image[40:60, 140:170] = colors['city']
    image[180:200, 50:70] = colors['city']

    # Add roads (thin lines)
    image[120:125, :] = colors['road']
    image[:, 150:155] = colors['road']

    return image


def visualize_layers(image: np.ndarray, layers: list, colors: np.ndarray, max_display: int = 8):
    """
    Visualize the binary layers of a decomposed image.

    Args:
        image: Original RGB image
        layers: List of binary layers
        colors: Array of colors corresponding to layers
        max_display: Maximum number of layers to display
    """
    num_layers = min(len(layers), max_display)

    fig, axes = plt.subplots(2, (num_layers + 2) // 2, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_layers):
        axes[i].imshow(layers[i], cmap='gray')
        color_hex = '#{:02x}{:02x}{:02x}'.format(*colors[i])
        axes[i].set_title(f'Layer {i}\nColor: {color_hex}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('output/layer_visualization.png', dpi=150, bbox_inches='tight')
    print("Layer visualization saved to output/layer_visualization.png")


def compare_filters_visual(original: np.ndarray,
                           noisy: np.ndarray,
                           filtered: np.ndarray,
                           filename: str = 'comparison.png'):
    """
    Create a visual comparison of original, noisy, and filtered images.

    Args:
        original: Original clean image
        noisy: Noisy image
        filtered: Filtered image
        filename: Output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(noisy)
    axes[1].set_title('Noisy')
    axes[1].axis('off')

    axes[2].imshow(filtered)
    axes[2].set_title('Filtered')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'output/{filename}', dpi=150, bbox_inches='tight')
    print(f"Comparison saved to output/{filename}")


def example_basic_filtering():
    """
    Example 1: Basic filtering with synthetic map.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Filtering")
    print("="*60)

    # Create output directory
    Path('output').mkdir(exist_ok=True)

    # Create synthetic map
    print("Creating synthetic map image...")
    # original = create_synthetic_map(256)
    original = load_image("input.png")

    # Get layer information
    colors = get_unique_colors(original)
    print(f"Image has {len(colors)} unique colors")

    # Quantize if needed (for fair evaluation)
    from src.quantization import should_quantize, quantize_colors
    if should_quantize(original, threshold=50):
        print(f"Quantizing image to discrete color palette...")
        original, palette = quantize_colors(original, n_colors=32)
        print(f"Quantized to {len(palette)} colors")

    # Add noise
    print("Adding impulsive noise (10%)...")
    noisy = inject_impulsive_noise(original, noise_ratio=0.10)

    # Apply filter
    print("Applying multi-layer filter...")
    mlf = MultiLayerFilter(
        filter_type='morphological',
        filter_params={'operation': 'opening', 'kernel_size': 3},
        use_segmentation=True,
        auto_quantize=False  # Already quantized
    )
    filtered = mlf.filter_image(noisy)

    # Evaluate
    print("Evaluating results...")
    metrics = evaluate_filter(original, noisy, filtered)
    print_evaluation_results(metrics, "Multi-Layer Filter")

    # Visualize
    print("Creating visualizations...")
    compare_filters_visual(original, noisy, filtered, 'basic_filtering_comparison.png')

    # Save images
    save_image(original, 'output/original.png')
    save_image(noisy, 'output/noisy.png')
    save_image(filtered, 'output/filtered.png')
    print("Images saved to output/ directory")


def example_layer_visualization():
    """
    Example 2: Visualize layer decomposition.
    """
    print("\n" + "="*60)
    print("Example 2: Layer Visualization")
    print("="*60)

    # Create output directory
    Path('output').mkdir(exist_ok=True)

    # Create synthetic map
    print("Creating synthetic map image...")
    # image = create_synthetic_map(256)
    image = load_image("input.png")

    # Decompose to layers
    from src.decomposition import decompose_to_layers
    print("Decomposing image to layers...")
    layers, colors, _ = decompose_to_layers(image, auto_quantize=True, quantize_colors=32, quantize_threshold=50)

    print(f"Decomposed into {len(layers)} layers")
    for i, color in enumerate(colors):
        pixel_count = np.sum(layers[i] > 0)
        print(f"  Layer {i}: Color RGB{tuple(color)}, {pixel_count} pixels")

    # Visualize layers
    visualize_layers(image, layers, colors)


def example_noise_comparison():
    """
    Example 3: Compare different noise types and filters.
    """
    print("\n" + "="*60)
    print("Example 3: Noise Type Comparison")
    print("="*60)

    # Create output directory
    Path('output').mkdir(exist_ok=True)

    # Create synthetic map
    print("Creating synthetic map image...")
    # original = create_synthetic_map(256)
    original = load_image("input.png")

    # Quantize if needed (for fair evaluation)
    from src.quantization import should_quantize, quantize_colors
    if should_quantize(original, threshold=50):
        print(f"Quantizing image to discrete color palette...")
        original, palette = quantize_colors(original, n_colors=32)
        print(f"Quantized to {len(palette)} colors")

    colors = get_unique_colors(original)

    # Test different noise types
    noise_types = [
        ('impulsive', lambda img: inject_impulsive_noise(img, 0.10)),
        ('content_dependent', lambda img: inject_content_dependent_noise(img, colors, 0.10))
    ]

    for noise_name, noise_func in noise_types:
        print(f"\nTesting {noise_name} noise...")

        # Add noise
        noisy = noise_func(original)

        # Apply filter
        mlf = MultiLayerFilter(
            filter_type='morphological',
            filter_params={'operation': 'opening', 'kernel_size': 3},
            use_segmentation=True,
            auto_quantize=False  # Already quantized
        )
        filtered = mlf.filter_image(noisy)

        # Evaluate
        metrics = evaluate_filter(original, noisy, filtered)
        print_evaluation_results(metrics, f"Multi-Layer Filter ({noise_name} noise)")

        # Visualize
        compare_filters_visual(
            original, noisy, filtered,
            f'comparison_{noise_name}_noise.png'
        )


def example_filter_comparison():
    """
    Example 4: Compare different filter configurations.
    """
    print("\n" + "="*60)
    print("Example 4: Filter Configuration Comparison")
    print("="*60)

    # Create output directory
    Path('output').mkdir(exist_ok=True)

    # Create synthetic map
    print("Creating synthetic map image...")
    # original = create_synthetic_map(256)
    original = load_image("input.png")

    # Quantize if needed (for fair evaluation)
    from src.quantization import should_quantize, quantize_colors
    if should_quantize(original, threshold=50):
        print(f"Quantizing image to discrete color palette...")
        original, palette = quantize_colors(original, n_colors=32)
        print(f"Quantized to {len(palette)} colors")

    # Add noise
    print("Adding noise...")
    noisy = inject_impulsive_noise(original, noise_ratio=0.10)

    # Test different filter configurations
    configs = [
        ('Morphological Opening', 'morphological', {'operation': 'opening', 'kernel_size': 3}),
        ('Morphological Closing', 'morphological', {'operation': 'closing', 'kernel_size': 3}),
        ('Median Filter', 'median', {'kernel_size': 3}),
        ('Combined Filter', 'combined', {})
    ]

    results = []

    for name, filter_type, params in configs:
        print(f"\nTesting {name}...")

        mlf = MultiLayerFilter(
            filter_type=filter_type,
            filter_params=params,
            use_segmentation=True,
            auto_quantize=False  # Already quantized
        )
        filtered = mlf.filter_image(noisy)

        metrics = evaluate_filter(original, noisy, filtered)
        results.append((name, metrics))

        print(f"{name}: ΔE improvement = {metrics['delta_e_improvement']:.2f}")

    # Print comparison
    print("\n" + "="*60)
    print("Filter Comparison Summary")
    print("="*60)
    print(f"{'Filter':<30} {'ΔE Improv.':<12} {'PSNR Improv.':<12}")
    print("-" * 60)
    for name, metrics in results:
        print(f"{name:<30} {metrics['delta_e_improvement']:>10.2f}  {metrics['psnr_improvement']:>10.2f}")


def main():
    """
    Run all examples.
    """
    print("\n" + "="*60)
    print("Multi-Layer Map Filter - Examples")
    print("="*60)

    try:
        example_basic_filtering()
        example_layer_visualization()
        example_noise_comparison()
        example_filter_comparison()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Check the 'output/' directory for results.")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

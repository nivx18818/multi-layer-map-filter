"""
Simple test script to verify the implementation works.
"""

import numpy as np
import sys

print("Testing multi-layer map filter implementation...")
print("=" * 60)

try:
    # Test imports
    print("\n1. Testing imports...")
    from src.decomposition import get_unique_colors, decompose_to_layers
    from src.filtering import MorphologicalFilter, MedianFilter
    from src.segmentation import segment_layers
    from src.reconstruction import reconstruct_image
    from src.evaluation import inject_impulsive_noise, evaluate_filter
    from src.main import MultiLayerFilter
    print("   ✓ All imports successful")

    # Create a simple test image
    print("\n2. Creating test image...")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:50, :50] = [255, 0, 0]    # Red
    image[:50, 50:] = [0, 255, 0]    # Green
    image[50:, :50] = [0, 0, 255]    # Blue
    image[50:, 50:] = [255, 255, 0]  # Yellow
    print(f"   ✓ Created {image.shape} test image")

    # Test decomposition
    print("\n3. Testing decomposition...")
    colors = get_unique_colors(image)
    print(f"   ✓ Found {len(colors)} unique colors")

    layers, colors = decompose_to_layers(image)
    print(f"   ✓ Decomposed into {len(layers)} layers")

    # Test filtering
    print("\n4. Testing filtering...")
    morph_filter = MorphologicalFilter(operation='opening', kernel_size=3)
    filtered_layer = morph_filter.apply(layers[0])
    print(f"   ✓ Applied morphological filter")

    median_filter = MedianFilter(kernel_size=3)
    filtered_layer = median_filter.apply(layers[0])
    print(f"   ✓ Applied median filter")

    # Test noise injection
    print("\n5. Testing noise injection...")
    noisy = inject_impulsive_noise(image, noise_ratio=0.05)
    print(f"   ✓ Added noise to image")

    # Test main pipeline
    print("\n6. Testing main pipeline...")
    mlf = MultiLayerFilter(
        filter_type='morphological',
        filter_params={'operation': 'opening', 'kernel_size': 3},
        use_segmentation=False  # Disable segmentation for quick test
    )
    filtered = mlf.filter_image(noisy)
    print(f"   ✓ Filtered noisy image")
    print(f"   ✓ Output shape: {filtered.shape}")

    # Test evaluation
    print("\n7. Testing evaluation...")
    metrics = evaluate_filter(image, noisy, filtered)
    print(f"   ✓ ΔE improvement: {metrics['delta_e_improvement']:.2f}")
    print(f"   ✓ PSNR improvement: {metrics['psnr_improvement']:.2f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

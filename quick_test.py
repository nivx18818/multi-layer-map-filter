"""
Quick test that writes output to file.
"""
import sys
import numpy as np

# Redirect stdout to file
with open('test_results.txt', 'w') as f:
    sys.stdout = f

    print("="*60)
    print("Multi-Layer Map Filter - Quick Test")
    print("="*60)

    try:
        # Test 1: Imports
        print("\n1. Testing imports...")
        from src.main import MultiLayerFilter
        from src.evaluation import inject_impulsive_noise, evaluate_filter
        print("   SUCCESS: All imports work")

        # Test 2: Create test image
        print("\n2. Creating test image...")
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[:25, :25] = [255, 0, 0]
        image[:25, 25:] = [0, 255, 0]
        image[25:, :] = [0, 0, 255]
        print(f"   SUCCESS: Created {image.shape} image")

        # Test 3: Add noise
        print("\n3. Adding noise...")
        noisy = inject_impulsive_noise(image, noise_ratio=0.05)
        print("   SUCCESS: Noise added")

        # Test 4: Filter
        print("\n4. Filtering...")
        mlf = MultiLayerFilter(
            filter_type='morphological',
            filter_params={'operation': 'opening', 'kernel_size': 3},
            use_segmentation=False
        )
        filtered = mlf.filter_image(noisy)
        print(f"   SUCCESS: Filtered image shape {filtered.shape}")

        # Test 5: Evaluate
        print("\n5. Evaluating...")
        metrics = evaluate_filter(image, noisy, filtered)
        print(f"   SUCCESS: Delta E improvement = {metrics['delta_e_improvement']:.2f}")
        print(f"   SUCCESS: PSNR improvement = {metrics['psnr_improvement']:.2f}")

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

sys.stdout = sys.__stdout__
print("Test results written to test_results.txt")

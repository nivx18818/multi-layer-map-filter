# Known Issues

## 1. Layer Visualization with High-Color Images ✅ FIXED

**Issue**: When `input.png` has many colors (2033+), layer visualization showed all-black layers.

**Root Cause**: `example_layer_visualization()` was using `auto_quantize=False`, which created 2033 layers with only 1-4 pixels each. These tiny layers appeared black when visualized.

**Solution**: Changed to `auto_quantize=True` to quantize images with >50 colors down to ~32 colors before decomposition.

## 2. Segmentation-Based Reconstruction Issues ⚠️ ONGOING

**Issue**: When `use_segmentation=True`, the filtered image has worse quality metrics than the noisy image (negative improvements).

**Root Cause**: The segmentation-based reconstruction algorithm assigns colors to regions incorrectly, changing ~33% of pixels instead of just the ~10% that are noisy.

**Current Status**:

- Segmentation-based reconstruction changes 32.81% of pixels incorrectly
- Simple reconstruction without segmentation gives 0% improvement (preserves noisy pixels)
- The algorithm works best with synthetic discrete-color maps but struggles with:
  - Real-world images with anti-aliasing
  - Palette-based noise distributed across all color layers
  - Small isolated features that look similar to noise

**Workaround**: Use `use_segmentation=False` for now, though this provides minimal filtering benefit.

**Needs Investigation**:

1. Region acceptance criteria (f1, f2 thresholds) may need tuning
2. Priority calculation strategy needs review
3. Distance transform for unlabeled pixels may have bugs
4. May need different morphological operations or kernel sizes

## 3. Noise Model Mismatch

**Issue**: The filtering algorithm doesn't improve metrics even with correct palette-based noise.

**Root Cause**: The algorithm expects noise to create ISOLATED pixels that morphological operations can remove. However:

- Palette-based noise distributes evenly across ALL color layers (~1900 pixels per layer)
- Morphological opening can't distinguish this from legitimate sparse features
- The noise doesn't look "isolated" from a per-layer perspective

**Implications**:

- Algorithm works best for synthetic maps with known structure
- Real-world map images may need pre-processing or different noise assumptions
- May need adaptive filtering based on local density rather than global morphology

## 4. Evaluation Metrics

**Issue**: All evaluation metrics show negative improvements (filtered worse than noisy).

**Status**: This is a consequence of issues #2 and #3 above. Once those are fixed, metrics should improve.

## Recommendations

1. For testing, use `create_synthetic_map()` which generates ideal discrete-color maps
2. For real images, ensure quantization is applied (`auto_quantize=True`)
3. Currently, set `use_segmentation=False` until reconstruction issues are resolved
4. Consider alternative noise models (e.g., clustered noise rather than random scattered pixels)
5. May need to implement the paper's full MUD (Morphologically-Invariant Universal Denoising) filter instead of basic morphological operations

## Next Steps

- [ ] Debug `reconstruct_with_regions()` to understand why regions get wrong colors
- [ ] Review segmentation acceptance criteria and tuning
- [ ] Implement proper priority-based reconstruction
- [ ] Test with synthetic maps to validate algorithm correctness
- [ ] Consider implementing adaptive local filters instead of global morphological operations

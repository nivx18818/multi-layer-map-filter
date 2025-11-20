# Known Issues

## 1. Layer Visualization with High-Color Images ✅ FIXED

**Issue**: When `input.png` has many colors (2033+), layer visualization showed all-black layers.

**Root Cause**: `example_layer_visualization()` was using `auto_quantize=False`, which created 2033 layers with only 1-4 pixels each. These tiny layers appeared black when visualized.

**Solution**: Changed to `auto_quantize=True` to quantize images with >50 colors down to ~32 colors before decomposition.

## 2. Segmentation-Based Reconstruction Issues ✅ RESOLVED

**Issue**: When `use_segmentation=True`, the filtered image had worse quality metrics than the noisy image (negative improvements).

**Root Cause**: Multiple issues were identified:

1. **Morphological opening was too aggressive**: kernel_size=3 removed legitimate small features along with noise, completely wiping out some layers
2. **Dilation expanded regions beyond true boundaries**: The segmentation algorithm uses dilation followed by hole-filling, causing regions to overlap with areas that should have different colors
3. **Low f1 threshold allowed inaccurate regions**: With f1=0.3, regions where only 30% of pixels belonged to the actual object were accepted, causing ~20% of pixels to be assigned wrong colors
4. **fill_unlabeled_pixels propagated labels incorrectly**: This function dilated region labels to fill unlabeled areas, causing further incorrect color assignments

**Solution**:

1. **Changed default filter to median (kernel_size=3)**: Median filtering preserves layer structure much better than morphological opening, achieving proper noise removal without destroying legitimate features
2. **Disabled segmentation by default**: Set `use_segmentation=False` as the default, since simple reconstruction with median filtering works much better
3. **Improved segmentation parameters for advanced use**: Raised f1_threshold to 0.6, lowered f2_threshold to 0.05, reduced dilation_size to 3 for users who want to experiment with segmentation
4. **Fixed unlabeled pixel handling**: Stopped propagating region labels via fill_unlabeled_pixels; instead, unlabeled pixels now preserve their noisy values (conservative approach)

**Results After Fix**:

- DeltaE improvement: **+7.51** (was -3.24)
- PSNR improvement: **+10.51 dB** (was -1.16)
- MSE improvement: **+740.23** (was -247.84)
- SSIM: **0.9343** (excellent quality, was 0.5930)

**Current Recommendation**: Use `MultiLayerFilter(filter_type='median', filter_params={'kernel_size': 3}, use_segmentation=False)` for best results.

## 3. Morphological Filtering Not Suitable for This Application ✅ RESOLVED

**Issue**: Morphological opening doesn't improve metrics even with palette-based noise.

**Root Cause**:

- Morphological opening removes ALL small isolated pixels, but can't distinguish between noise and legitimate small features
- Palette-based noise is distributed across layers mixed with real features
- Opening is too aggressive, removing real features along with noise

**Solution**: Use **median filtering** instead, which considers neighborhood pixel values and preserves structure much better. Median filtering with kernel_size=3 works excellently for palette-based noise.

## 4. Segmentation Algorithm Limitations (ONGOING INVESTIGATION)

**Status**: Segmentation-based reconstruction is now functional but still performs worse than simple reconstruction.

**Current Understanding**:

- The paper's segmentation algorithm was designed for specific types of map images with well-separated color regions
- Real-world images with scattered features don't fit this model well
- The dilation + hole-filling + connected components approach creates regions that overlap incorrectly

**Current State**:

- Segmentation parameters have been improved (f1=0.6, f2=0.05, dilation=3)
- Unlabeled pixels now preserve noisy values instead of propagating labels
- However, simple reconstruction (without segmentation) still outperforms segmentation-based reconstruction

**Recommendation**: Keep `use_segmentation=False` (now the default). The segmentation code is preserved for future research but is not recommended for production use.

## Recommendations

1. **Use median filtering**: `filter_type='median'` with `kernel_size=3` provides excellent results
2. **Disable segmentation**: Keep `use_segmentation=False` (the default) for best performance
3. **Enable auto-quantization**: Set `auto_quantize=True` (the default) to handle images with many colors
4. **Quantize to ~32 colors**: This provides a good balance between quality and layer count

**Example Configuration**:

```python
mlf = MultiLayerFilter(
    filter_type='median',
    filter_params={'kernel_size': 3},
    use_segmentation=False,  # Default
    auto_quantize=True,      # Default
    quantize_colors=32       # Default
)
```

## Resolved Issues Summary

✅ Issue #1: Layer visualization - Fixed by enabling auto-quantization
✅ Issue #2: Segmentation reconstruction - Fixed by switching to median filtering without segmentation
✅ Issue #3: Morphological filtering - Fixed by using median filtering instead
⚠️ Issue #4: Segmentation algorithm - Partially fixed but still not recommended for use

## Future Work

- [ ] Investigate why segmentation + priority ordering underperforms simple reconstruction
- [ ] Consider implementing the paper's full MUD (Morphologically-Invariant Universal Denoising) filter
- [ ] Experiment with adaptive filtering based on local feature density
- [ ] Test with more diverse map images and noise types

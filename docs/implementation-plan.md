## Plan: Multi-layer Filtering for Raster Map Images

This plan outlines implementing the ICIP 2009 paper's novel approach that decomposes color map images into binary layers, filters each layer independently, then reconstructs the image using region-based color priority ordering.

### Steps

1. **Implement color-to-layer decomposition** - Create functions to extract unique colors from input image, generate N binary layers where each layer L_t(i,j)=1 only for pixels matching color t, and store layers as list of binary masks using NumPy arrays.

2. **Apply binary image filtering to each layer** - Implement morphological operations (dilation, erosion, opening, closing) using OpenCV functions like `cv2.dilate()` and `cv2.erode()`, apply median filtering on binary layers, and support pluggable filter types (basic morphological or advanced methods).

3. **Build region-based segmentation algorithm** - Follow Figure 3 pseudocode to segment filtered layers into distinct regions using dilation + hole-filling + connected components via `cv2.connectedComponents()`, calculate acceptance criteria (f1: object pixel ratio, f2: labeled pixel percentage) for each candidate region, and generate label mask S_M assigning each pixel to a region.

4. **Implement local color priority ordering** - For each segmented region, calculate color frequency distribution, assign priorities (lowest frequency = highest priority) to preserve fine details over backgrounds, handle distance transform via `cv2.distanceTransform()` for non-labeled pixels to assign nearest region label.

5. **Merge filtered layers back to color image** - Iterate through pixels, select output color from highest-priority non-zero layer per equation (3), reconstruct final filtered RGB image from binary layer stack.

6. **Create evaluation and testing framework** - Implement noise injection (impulsive and content-dependent), calculate mean color distance (ΔE) in L*a*b space using `cv2.cvtColor()` for color space conversion, compare against baseline filters (adaptive vector median, morphological, peer group).

### Further Considerations

1. **Binary filter selection** - Should we prioritize spatially-variant morphology or discrete universal denoising? The paper tests both (MSM and MUD). Recommend starting with standard morphological operations for simplicity.

2. **Layer ordering strategy** - The paper presents both global (frequency-based) and region-based ordering. Should we implement both and make it configurable? Region-based is more complex but handles heterogeneous backgrounds better.

3. **Performance optimization** - Processing N layers separately could be slow for images with many colors. Consider parallel processing of independent layers using multiprocessing or vectorized NumPy operations where possible?

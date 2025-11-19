# Multi-Layer Map Filter

A Python implementation of the multi-layer filtering approach for color map images based on the ICIP 2009 paper. This method decomposes color map images into binary layers, filters each layer independently, and reconstructs the image using region-based color priority ordering.

## Features

- **Layer Decomposition**: Extracts unique colors and creates binary layers
- **Binary Filtering**: Supports morphological operations (dilation, erosion, opening, closing) and median filtering
- **Region-Based Segmentation**: Implements the paper's algorithm for segmenting filtered layers
- **Color Priority Ordering**: Assigns priorities based on color frequency to preserve fine details
- **Evaluation Framework**: Includes noise injection and multiple quality metrics (ΔE, PSNR, MSE, SSIM)
- **Flexible Architecture**: Pluggable filter types and configurable parameters

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository
2. Navigate to the project directory
3. Create a virtual environment:

```powershell
python -m venv venv
```

4. Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

5. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.main import MultiLayerFilter, load_image, save_image
from src.evaluation import inject_impulsive_noise

# Load image
original = load_image('input.png')

# Add noise
noisy = inject_impulsive_noise(original, noise_ratio=0.10)

# Create filter
mlf = MultiLayerFilter(
    filter_type='morphological',
    filter_params={'operation': 'opening', 'kernel_size': 3},
    use_segmentation=True
)

# Apply filter
filtered = mlf.filter_image(noisy)

# Save result
save_image(filtered, 'output.png')
```

### Running Examples

The project includes comprehensive examples demonstrating different features:

```powershell
python example_usage.py
```

This will run all examples and save visualizations to the `output/` directory:

- Basic filtering with synthetic map
- Layer visualization
- Noise type comparison (impulsive vs content-dependent)
- Filter configuration comparison

### Filter Configuration

#### Morphological Filter

```python
mlf = MultiLayerFilter(
    filter_type='morphological',
    filter_params={
        'operation': 'opening',  # or 'closing', 'dilation', 'erosion'
        'kernel_size': 3,        # odd number (3, 5, 7, etc.)
        'iterations': 1          # number of times to apply
    }
)
```

#### Median Filter

```python
mlf = MultiLayerFilter(
    filter_type='median',
    filter_params={
        'kernel_size': 3  # odd number
    }
)
```

#### Combined Filter

```python
mlf = MultiLayerFilter(
    filter_type='combined'
    # Uses median + morphological opening by default
)
```

### Evaluation

```python
from src.evaluation import evaluate_filter, print_evaluation_results

# Evaluate filter performance
metrics = evaluate_filter(original, noisy, filtered)
print_evaluation_results(metrics, "My Filter")
```

Metrics include:

- **ΔE**: Mean color distance in CIE L*a*b\* space
- **PSNR**: Peak Signal-to-Noise Ratio
- **MSE**: Mean Squared Error
- **SSIM**: Structural Similarity Index

## Project Structure

```
multi-layer-map-filter/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── decomposition.py      # Color-to-layer decomposition
│   ├── filtering.py          # Binary image filtering
│   ├── segmentation.py       # Region-based segmentation
│   ├── reconstruction.py     # Layer merging and reconstruction
│   ├── evaluation.py         # Evaluation metrics and noise injection
│   └── main.py              # Main pipeline and convenience functions
├── docs/
│   └── implementation-plan.md # Detailed implementation plan
├── example_usage.py          # Example scripts
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── venv/                    # Virtual environment (created during setup)
```

## Algorithm Overview

The multi-layer filtering approach consists of five main steps:

1. **Decomposition**: Extract unique colors and create binary layers where each layer contains pixels of one color
2. **Filtering**: Apply binary filters (morphological, median, etc.) to each layer independently
3. **Segmentation**: Use dilation, hole-filling, and connected components to segment layers into regions
4. **Prioritization**: Calculate color priorities based on frequency (rare colors = higher priority)
5. **Reconstruction**: Merge filtered layers using priority ordering to create the final image

## Dependencies

- **numpy**: Array operations and numerical computing
- **opencv-python**: Image processing and morphological operations
- **scikit-image**: Additional image processing utilities
- **matplotlib**: Visualization and result plotting

## Performance Considerations

- **Number of colors**: Images with many unique colors will have more layers to process. Consider color quantization for very diverse images.
- **Image size**: Large images may require significant memory. Process in tiles if needed.
- **Parallelization**: Layers can be filtered independently, allowing for parallel processing.

## Configuration Options

### Segmentation Parameters

```python
mlf = MultiLayerFilter(
    use_segmentation=True,
    f1_threshold=0.5,  # Object pixel ratio threshold
    f2_threshold=0.8   # Labeled pixel percentage threshold
)
```

### Filter Selection Guidelines

- **Opening**: Removes small bright spots (noise), smooths boundaries
- **Closing**: Fills small dark spots (holes), connects nearby objects
- **Median**: Good for salt-and-pepper noise, preserves edges
- **Combined**: Robust for multiple noise types

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:

1. Virtual environment is activated
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're running from the project root directory

### Memory Issues

For large images:

- Reduce image size before processing
- Process fewer layers at once
- Use simpler filters (smaller kernel sizes)

## Future Enhancements

Potential improvements from the implementation plan:

1. **Binary filter selection**: Add spatially-variant morphology or discrete universal denoising
2. **Global vs region-based ordering**: Implement both strategies as configurable options
3. **Performance optimization**: Parallel processing of independent layers
4. **Baseline comparisons**: Add adaptive vector median and peer group filters

## References

Based on the ICIP 2009 paper on multi-layer filtering for color map images.

## License

This implementation is provided for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

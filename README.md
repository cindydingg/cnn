# CNN Research Project

## Overview
This project implements a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset. It covers CNN layer development, forward and backward propagation, batch normalization, and training optimizations to improve validation accuracy.

## Features
- Implementation of convolutional, pooling, and fully connected layers.
- Forward and backward propagation for CNN layers.
- Spatial batch normalization for improved training efficiency.
- Performance comparison between naive and optimized implementations.
- Training a three-layer CNN to achieve >65% validation accuracy on CIFAR-10.

## Project Structure
```
|-- CNN-BatchNorm.ipynb        # Jupyter notebook for batch normalization in CNNs
|-- CNN-Layers.ipynb           # Jupyter notebook for implementing CNN layers
|-- CNN.ipynb                  # Main Jupyter notebook for running CNN experiments
|-- README.md                  # Project documentation
|-- __init__.py                # Init file for Python module
|-- cnn.py                      # Core CNN implementation
|-- conv_layer_utils.py         # Utility functions for CNN layers
|-- conv_layers.py              # Convolutional layers implementation
|-- data_utils.py               # Data loading and preprocessing functions
|-- fast_layers.py              # Optimized CNN layer implementations
|-- features.py                 # Feature extraction and handling
|-- gradient_check.py           # Gradient checking utility
|-- im2col.py                   # Image-to-column conversion function
|-- im2col_cython.c             # Cython implementation for im2col
|-- im2col_cython.cpython-311-darwin.so  # Compiled Cython shared object (macOS)
|-- im2col_cython.cpython-36m-darwin.so  # Compiled Cython shared object (macOS)
|-- im2col_cython.pyx           # Cython source code for im2col
|-- layer_tests.py              # Unit tests for layers
|-- layer_utils.py              # General utility functions for layers
|-- layers.py                   # Fully connected and CNN layer functions
|-- optim.py                    # Optimization functions
|-- setup.py                    # Setup script for the project
|-- solver.py                   # Solver for training the CNN
|-- vis_utils.py                 # Visualization utilities
```

## Dependencies
- Python 3.x
- NumPy
- Matplotlib

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cnn-research.git
   cd cnn-research
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running Jupyter Notebooks
To explore different CNN components, open Jupyter and run:
```bash
jupyter notebook
```

Then, navigate to the appropriate `.ipynb` file.

### Training the CNN
To train the three-layer CNN on CIFAR-10:
```bash
python solver.py
```

### Expected Output
During training, the loss should decrease, and validation accuracy should improve. A well-tuned model should achieve >65% validation accuracy on CIFAR-10.

### Model Architecture
```
conv - relu - 2x2 max pool - affine - relu - affine - softmax
```

### Performance Comparison
The project includes implementations for both naive and optimized CNN layers. The optimized layers significantly reduce computation time while maintaining accuracy.

## Results
- Overfitting a small dataset successfully.
- Achieved >65% validation accuracy on CIFAR-10 with optimized hyperparameters.
- Significant speedup using optimized CNN layers, achieving up to **17x faster forward pass** and **840x faster backward pass** compared to naive implementations.
- The naive implementation of CNN layers resulted in higher computational costs, whereas the optimized versions significantly reduced runtime while maintaining accuracy.
- Batch normalization improved training stability and convergence speed, leading to a more efficient training process.
- The use of Adam optimizer with tuned learning rates led to better generalization and improved validation accuracy over multiple training runs.
- Final trained model was able to achieve consistent classification accuracy across different CIFAR-10 test batches, demonstrating robustness of the model architecture.

## Future Improvements
- Experiment with deeper architectures (e.g., ResNet, VGGNet).
- Implement data augmentation techniques.
- Fine-tune hyperparameters for further accuracy improvements.

## Credits
- Project inspired by Stanford CS231n assignments.
- Code adapted from Justin Johnson & Serena Yeung.

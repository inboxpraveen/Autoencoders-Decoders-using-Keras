# Autoencoders for Image Reconstruction and Generation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive, educational implementation of various autoencoder architectures for image processing. This project demonstrates dimensionality reduction, image reconstruction, denoising, similarity search, and image morphing using deep learning.

## 🎯 Project Overview

This project implements three types of autoencoders with increasing complexity:

1. **PCA Autoencoder** - Linear dimensionality reduction similar to Principal Component Analysis
2. **Convolutional Autoencoder** - Deep CNN-based encoder-decoder for better image reconstruction
3. **Denoising Autoencoder** - Learns robust features by reconstructing clean images from noisy inputs

Additionally, the project includes:
- **Image Retrieval** - Find similar images using learned latent representations
- **Image Morphing** - Smooth interpolation between images in latent space

## 🏗️ Architecture

### PCA Autoencoder
```
Input → Flatten → Dense(code_size) → Dense(original_size) → Reshape → Output
```
A simple linear autoencoder that compresses images into a low-dimensional code, similar to PCA but learned through backpropagation.

### Convolutional Autoencoder
```
Encoder: Conv2D + MaxPool (×4) → Flatten → Dense(code_size)
Decoder: Dense → Reshape → Conv2DTranspose (×4)
```
Uses convolutional layers to learn hierarchical features and transpose convolutions to reconstruct images.

### Denoising Autoencoder
Same architecture as convolutional autoencoder, but trained with corrupted inputs and clean targets to learn robust, noise-invariant features.

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Autoencoders-Decoders-using-Keras.git
   cd Autoencoders-Decoders-using-Keras
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

The project provides a unified command-line interface through `main.py`:

### Training Models

#### 1. PCA Autoencoder
Train a simple linear autoencoder (quick, good for understanding basics):
```bash
python main.py --mode pca --epochs 15 --code-size 32
```

#### 2. Convolutional Autoencoder
Train a deep CNN autoencoder (best reconstruction quality):
```bash
python main.py --mode convolutional --epochs 25 --code-size 32 --batch-size 32
```

#### 3. Denoising Autoencoder
Train an autoencoder to remove noise from images:
```bash
python main.py --mode denoising --epochs 25 --code-size 512 --noise-sigma 0.1
```

### Using Trained Models

#### Image Retrieval
Find similar images using learned representations:
```bash
python main.py --mode retrieval --model-path saved_models --n-queries 5 --n-neighbors 5
```

#### Image Morphing
Create smooth transitions between images:
```bash
python main.py --mode morphing --model-path saved_models --n-pairs 5 --n-steps 7
```

## 📊 Command-Line Arguments

### General Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | **required** | Operation mode: pca, convolutional, denoising, retrieval, morphing |
| `--image-size` | int | 32 | Image dimensions (width/height) |
| `--test-size` | float | 0.1 | Fraction of data for testing |

### Model Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--code-size` | int | 32 | Latent code dimension (use 512 for denoising) |
| `--epochs` | int | 25 | Number of training epochs |
| `--batch-size` | int | 32 | Training batch size |

### Denoising Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--noise-sigma` | float | 0.1 | Gaussian noise standard deviation |

### Retrieval Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-queries` | int | 3 | Number of query images |
| `--n-neighbors` | int | 5 | Number of similar images to retrieve |

### Morphing Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-pairs` | int | 5 | Number of image pairs to morph |
| `--n-steps` | int | 7 | Interpolation steps |

### Path Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save-dir` | str | saved_models | Directory to save trained models |
| `--checkpoint-dir` | str | checkpoints | Directory for training checkpoints |
| `--model-path` | str | saved_models | Path to load trained models |

## 📁 Project Structure

```
Autoencoders-Decoders-using-Keras/
│
├── main.py                    # Main entry point with CLI
│
├── models/                    # Model architectures
│   ├── __init__.py
│   ├── pca_autoencoder.py
│   ├── convolutional_autoencoder.py
│   └── denoising_autoencoder.py
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── visualization.py      # Plotting and visualization
│   └── noise.py              # Noise generation utilities
│
├── image_retrieval.py        # Similarity search implementation
├── image_morphing.py         # Image interpolation
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🧠 Understanding Autoencoders

### What is an Autoencoder?

An autoencoder is a neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct the original data from this compressed form (decoding).

```
Original → [Encoder] → Compressed Code → [Decoder] → Reconstruction
```

### Why Use Autoencoders?

1. **Dimensionality Reduction** - Compress high-dimensional data
2. **Feature Learning** - Learn meaningful representations automatically
3. **Denoising** - Remove noise while preserving important features
4. **Anomaly Detection** - Identify unusual patterns
5. **Generative Modeling** - Create new, similar data

### Key Concepts

#### Latent Space (Code)
The compressed representation learned by the encoder. Points close together in latent space represent similar images.

#### Reconstruction Loss
The difference between input and output (typically Mean Squared Error). Lower loss means better reconstruction.

#### Bottleneck
The smallest layer (latent code) that forces the network to learn efficient representations.

## 📈 Results

After training, you'll find:

- **Training curves** - Loss over epochs showing model improvement
- **Reconstructions** - Original vs reconstructed images showing quality
- **Denoising examples** - Original → Noisy → Denoised progression
- **Similar images** - Query image with nearest neighbors
- **Morphing sequences** - Smooth transitions between image pairs

All results are saved in the `results/` directory.

## 🔬 Technical Details

### Dataset
The project uses the **Labeled Faces in the Wild (LFW)** dataset, which contains face images. The dataset is automatically downloaded via scikit-learn. If unavailable, synthetic data is generated for demonstration.

### Training Tips

1. **Start with PCA** - Quick training to verify setup
2. **Use GPU** - Significantly faster for convolutional models
3. **Monitor loss** - Should decrease steadily during training
4. **Adjust code size** - Smaller = more compression, larger = better quality
5. **Early stopping** - Training stops automatically if no improvement

### Model Checkpoints
Models are automatically saved during training. Best models are kept based on validation loss.

## 🛠️ Advanced Usage

### Custom Dataset
To use your own images, modify `utils/data_loader.py`:

```python
def load_custom_dataset(image_dir, img_size=32):
    # Load your images here
    images = []
    # ... your loading code ...
    return np.array(images)
```

### Hyperparameter Tuning
Experiment with different configurations:

```bash
# Larger latent code for better quality
python main.py --mode convolutional --code-size 128 --epochs 50

# Stronger denoising
python main.py --mode denoising --noise-sigma 0.2 --code-size 512
```

### Export Models
Trained models are saved in `saved_models/`:
- `encoder.weights.h5` - Encoder weights
- `decoder.weights.h5` - Decoder weights

## 📚 Educational Resources

### Understanding the Code

1. **Start with PCA** (`models/pca_autoencoder.py`) - Simple linear transformations
2. **Move to CNN** (`models/convolutional_autoencoder.py`) - Hierarchical features
3. **Explore denoising** (`models/denoising_autoencoder.py`) - Robust learning

### Key Learning Points

- **Encoder-Decoder architecture** - Symmetric compression and reconstruction
- **Transpose convolution** - Upsampling in the decoder
- **Latent space** - Learned feature representations
- **Transfer learning** - Encoder features useful for other tasks

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional autoencoder variants (VAE, β-VAE)
- More noise types (salt-and-pepper, blur)
- Different architectures (ResNet-based, U-Net)
- Additional applications (style transfer, inpainting)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original dataset: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
- Inspired by classical computer vision and deep learning research
- Built with TensorFlow/Keras

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Learning! 🚀**

*This project is designed to be educational, demonstrating key concepts in autoencoders while maintaining professional code quality and industry standards.*

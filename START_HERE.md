# 🚀 Quick Start Guide

Get started with autoencoders in 5 minutes with complete command reference!

---

## ⚡ Fastest Start (2 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Quick Example

```bash
python example_usage.py
```

**What happens:**
- ✅ Downloads face image dataset (LFW)
- ✅ Trains a simple PCA autoencoder (~2 minutes)
- ✅ Shows compression: 3,072 numbers → 32 numbers → reconstructed image
- ✅ Saves visualizations to `results/` folder

**Results:**
- `results/quick_example_samples.png` - Original images
- `results/quick_example_reconstruction.png` - Before & after compression

---

## 🎯 Complete Command Reference

### Training Modes

#### 1. PCA Autoencoder (Fast & Simple)

```bash
python main.py --mode pca --epochs 15 --code-size 32
```

**What it does:**
- Trains a simple linear autoencoder (like PCA)
- Fastest training (~2 minutes)
- Good for learning basics
- Compresses images using linear transformations

**When to use:**
- First time learning autoencoders
- Quick experiments
- Understanding dimensionality reduction
- Baseline comparison

**Output:**
- `results/pca/training_history.png` - Loss curves
- `results/pca/reconstruction_*.png` - Before/after images

---

#### 2. Convolutional Autoencoder (Best Quality)

```bash
python main.py --mode convolutional --epochs 25 --code-size 32
```

**What it does:**
- Trains a deep CNN autoencoder
- Best reconstruction quality (~20 minutes)
- Learns hierarchical features (edges → textures → faces)
- Production-ready architecture

**When to use:**
- Production applications
- Best image quality needed
- Learning deep learning concepts
- Building on top of autoencoders

**Output:**
- `results/convolutional/training_history.png` - Training progress
- `results/convolutional/reconstruction_*.png` - High-quality reconstructions
- `saved_models/encoder.weights.h5` - Trained encoder
- `saved_models/decoder.weights.h5` - Trained decoder

**Architecture:**
```
Input (32×32×3) 
  → Conv + Pool (4 layers) 
  → Code (32 numbers) 
  → Transpose Conv (4 layers) 
  → Output (32×32×3)
```

---

#### 3. Denoising Autoencoder (Advanced)

```bash
python main.py --mode denoising --epochs 25 --code-size 512 --noise-sigma 0.1
```

**What it does:**
- Trains to remove noise from images (~40 minutes)
- Adds Gaussian noise to inputs, learns to recover clean images
- Learns robust, noise-invariant features
- Uses larger latent space (512) for better quality

**When to use:**
- Image restoration tasks
- Noisy data
- Learning robust features
- Advanced applications

**Parameters:**
- `--noise-sigma 0.1` - Standard deviation of noise (0.1 = mild, 0.5 = heavy)
- `--code-size 512` - Larger for better denoising quality

**Output:**
- `results/denoising/training_history.png` - Training curves
- `results/denoising/denoising_*.png` - Original → Noisy → Cleaned
- Shows noise removal capability

---

### Application Modes

#### 4. Image Retrieval (Find Similar Images)

```bash
python main.py --mode retrieval --model-path saved_models --n-queries 3 --n-neighbors 5
```

**What it does:**
- Loads trained encoder
- Encodes all training images into latent codes
- For each query, finds most similar images using nearest neighbors
- Visualizes query + similar images

**How it works:**
1. Encode all images → 32-number codes
2. Similar images have similar codes
3. Use k-nearest neighbors to find matches
4. Display results side-by-side

**When to use:**
- Image search engines
- Finding similar faces/objects
- Content-based retrieval
- Understanding latent space

**Parameters:**
- `--n-queries 3` - Number of test images to query
- `--n-neighbors 5` - How many similar images to retrieve

**Output:**
- `results/retrieval/retrieval_query_*.png` - Query + matches with distances

---

#### 5. Image Morphing (Smooth Transitions)

```bash
python main.py --mode morphing --model-path saved_models --n-pairs 5 --n-steps 7
```

**What it does:**
- Loads trained encoder + decoder
- Takes two random images
- Interpolates between their latent codes
- Decodes intermediate codes → smooth transition

**How it works:**
1. Encode image A → code A
2. Encode image B → code B
3. Create intermediate codes: 0.8×A + 0.2×B, 0.6×A + 0.4×B, etc.
4. Decode each intermediate code → morphed image
5. Result: A → transition images → B

**When to use:**
- Creative effects
- Animation generation
- Understanding latent space structure
- Data augmentation

**Parameters:**
- `--n-pairs 5` - Number of image pairs to morph
- `--n-steps 7` - Interpolation steps (more = smoother)

**Output:**
- `results/morphing/morphing_pair_*.png` - Smooth transitions between faces

---

## 🎛️ Customization Options

### Common Parameters

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--epochs` | 25 | Training iterations | `--epochs 50` |
| `--batch-size` | 32 | Samples per batch | `--batch-size 16` |
| `--code-size` | 32 | Latent dimension | `--code-size 128` |
| `--image-size` | 32 | Image width/height | `--image-size 64` |

### Experiment Examples

```bash
# Longer training for better results
python main.py --mode convolutional --epochs 50

# Larger latent space (less compression, better quality)
python main.py --mode convolutional --code-size 128

# Smaller batch size (less memory, more stable)
python main.py --mode convolutional --batch-size 16

# Heavy noise removal
python main.py --mode denoising --noise-sigma 0.2

# More morphing steps (smoother transitions)
python main.py --mode morphing --n-steps 15

# Higher resolution images
python main.py --mode convolutional --image-size 64
```

---

## 📊 Expected Results

### Training Times (Approximate)

| Model | CPU | GPU | Epochs | Parameters |
|-------|-----|-----|--------|------------|
| PCA | 2 min | 1 min | 15 | ~100K |
| Convolutional | 20 min | 5 min | 25 | ~800K |
| Denoising | 40 min | 10 min | 25 | ~800K |

*Based on 12K images, 32×32 resolution*

### Quality Metrics

| Model | Code Size | MSE Loss | Use Case |
|-------|-----------|----------|----------|
| PCA | 32 | ~0.006 | Learning |
| Conv-32 | 32 | ~0.005 | Production |
| Conv-128 | 128 | ~0.003 | High quality |
| Denoising | 512 | ~0.006 | Noise removal |

---

## 🔍 Understanding the Output

### During Training

```
Epoch 1/25
loss: 0.0234 - mae: 0.1123 - val_loss: 0.0198
```

- `loss` - Training error (lower = better)
- `val_loss` - Test error (should track training)
- Watch for: Steadily decreasing values

### Result Files

**Training History**
- `training_history.png` - Loss curves over epochs
- Should show smooth decrease
- Validation should follow training

**Reconstructions**
- `reconstruction_*.png` - Original | Latent Code | Reconstructed
- Compare input vs output quality
- Code visualization shows learned representation

**Denoising**
- `denoising_*.png` - Clean | Noisy | Denoised
- Shows noise removal capability
- Cleaner output = better model

**Retrieval**
- `retrieval_query_*.png` - Query image + N similar matches
- Shows distances in latent space
- Closer images = more similar

**Morphing**
- `morphing_pair_*.png` - Smooth transition sequence
- α = 0.0 (image A) → α = 1.0 (image B)
- Should show gradual, realistic transitions

---

## 🛠️ Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
python main.py --mode convolutional --batch-size 16

# Or use smaller images
python main.py --mode convolutional --image-size 32
```

### Slow Training

```bash
# Fewer epochs
python main.py --mode pca --epochs 10

# Or try PCA mode first
python main.py --mode pca
```

### Poor Reconstruction Quality

```bash
# Train longer
python main.py --mode convolutional --epochs 50

# Or larger latent code
python main.py --mode convolutional --code-size 128
```

### Model Not Found (retrieval/morphing)

```bash
# Train a model first
python main.py --mode convolutional --epochs 25

# Then use it
python main.py --mode retrieval --model-path saved_models
```

---

## 📁 Output Structure

After running commands, you'll have:

```
Autoencoders-Decoders-using-Keras/
├── results/                      # All visualizations
│   ├── sample_images.png        # Dataset samples
│   ├── pca/                     # PCA results
│   ├── convolutional/           # Conv results
│   ├── denoising/               # Denoising results
│   ├── retrieval/               # Similar images
│   └── morphing/                # Transitions
│
├── saved_models/                # Trained weights
│   ├── encoder.weights.h5
│   └── decoder.weights.h5
│
├── checkpoints/                 # Training checkpoints
│   └── *.keras                  # Best models
│
└── logs/                        # TensorBoard logs
    └── convolutional/
```

---

## 🎓 Learning Path

### Beginner (Day 1)
```bash
# 1. Quick test
python example_usage.py

# 2. Simple model
python main.py --mode pca --epochs 15

# Understand: What is an autoencoder?
```

### Intermediate (Day 2-3)
```bash
# 1. Better model
python main.py --mode convolutional --epochs 25

# 2. Find similar images
python main.py --mode retrieval --model-path saved_models

# Understand: How do CNNs learn features?
```

### Advanced (Day 4-5)
```bash
# 1. Denoising
python main.py --mode denoising --epochs 25 --code-size 512

# 2. Creative morphing
python main.py --mode morphing --model-path saved_models --n-steps 10

# Understand: Latent space interpolation
```

---

## 💡 Pro Tips

### Best Practices

1. **Start Simple**: Always try PCA mode first to verify setup
2. **Monitor Training**: Watch loss decrease - should be smooth
3. **Save Often**: Models auto-save, but check `saved_models/`
4. **Experiment**: Try different `--code-size` values (16, 32, 64, 128, 256)
5. **Use GPU**: 5-10x faster if available (auto-detected)

### Hyperparameter Tuning

**Code Size Impact:**
- Small (16-32): Fast, compact, lower quality
- Medium (64-128): Balanced
- Large (256-512): Slow, best quality, for denoising

**Batch Size Impact:**
- Small (8-16): Less memory, more stable, slower
- Medium (32): Default, balanced
- Large (64-128): More memory, faster, less stable

**Epochs Impact:**
- Few (5-10): Quick test, underfitting
- Medium (25-30): Standard, good results
- Many (50+): Best quality, diminishing returns

---

## 🔗 Next Steps

### Read More
- **README.md** - Complete documentation with theory
- **PROJECT_SUMMARY.md** - Technical details, concepts, architecture
- **Code files** - Well-commented, educational

### Extend the Project
1. Try custom datasets (modify `utils/data_loader.py`)
2. Add new architectures (check `models/`)
3. Implement new noise types (`utils/noise.py`)
4. Create visualizations (`utils/visualization.py`)

### Get Help

```bash
# Show all options
python main.py --help

# Check project structure
ls models/
ls utils/
```

---

## ✨ Quick Command Summary

```bash
# Install
pip install -r requirements.txt

# Quick test (2 min)
python example_usage.py

# Train models
python main.py --mode pca --epochs 15                    # Fast
python main.py --mode convolutional --epochs 25          # Best
python main.py --mode denoising --code-size 512          # Advanced

# Applications
python main.py --mode retrieval --model-path saved_models
python main.py --mode morphing --model-path saved_models --n-steps 10

# Help
python main.py --help
```

---

**That's it! You're ready to explore autoencoders! 🎉**

For detailed theory and architecture, see **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**

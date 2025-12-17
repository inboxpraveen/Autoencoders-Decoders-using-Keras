# 🚀 Quick Start Guide

Get started with autoencoders in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run Quick Example

```bash
python example_usage.py
```

This trains a simple autoencoder in ~2 minutes and saves results to `results/` folder.

## Step 3: Try Different Modes

### Train Models

```bash
# Fast - PCA autoencoder (2 minutes)
python main.py --mode pca --epochs 15

# Best Quality - Convolutional autoencoder (20 minutes)
python main.py --mode convolutional --epochs 25

# Advanced - Denoising autoencoder (40 minutes)
python main.py --mode denoising --epochs 25 --code-size 512
```

### Use Trained Models

```bash
# Find similar images
python main.py --mode retrieval --model-path saved_models

# Create image morphing effects
python main.py --mode morphing --model-path saved_models
```

## Common Options

```bash
# Get help
python main.py --help

# Custom settings
python main.py --mode convolutional --epochs 50 --code-size 128 --batch-size 16
```

## Results

All outputs are saved to:
- `results/` - Visualizations
- `saved_models/` - Trained model weights
- `checkpoints/` - Training checkpoints

## Next Steps

- See **README.md** for complete documentation
- See **PROJECT_SUMMARY.md** for upgrade details
- Modify code in `models/` or `utils/` to customize

That's it! Enjoy exploring autoencoders! 🎉

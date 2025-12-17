"""Quick example demonstrating the autoencoder project.

This script provides a simple, self-contained example of how to use
the autoencoder models without command-line arguments.

Run this file to get a quick overview of the project capabilities.
"""

import os
import numpy as np
import tensorflow as tf
from utils import load_lfw_dataset, prepare_data
from models import build_pca_autoencoder
from utils.visualization import visualize_reconstruction, plot_sample_images

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create results directory
os.makedirs('results', exist_ok=True)

print("="*70)
print("AUTOENCODER PROJECT - QUICK EXAMPLE")
print("="*70)

# 1. Load and prepare data
print("\n1. Loading dataset...")
X, _ = load_lfw_dataset(dimx=32, dimy=32)
X_train, X_test = prepare_data(X, test_size=0.1)
img_shape = X_train.shape[1:]

print(f"   ✓ Loaded {len(X_train)} training images")
print(f"   ✓ Image shape: {img_shape}")

# 2. Show sample images
print("\n2. Displaying sample images...")
plot_sample_images(X_train[:6], title="Sample Images", save_path="results/quick_example_samples.png")
print("   ✓ Saved to: results/quick_example_samples.png")

# 3. Build and train PCA autoencoder (fast)
print("\n3. Training PCA Autoencoder (this is quick)...")
autoencoder, encoder, decoder = build_pca_autoencoder(img_shape, code_size=32)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X_train, X_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, X_test),
    verbose=0
)

test_loss = autoencoder.evaluate(X_test, X_test, verbose=0)
print(f"   ✓ PCA Autoencoder trained! Test MSE: {test_loss:.6f}")

# 4. Visualize reconstruction
print("\n4. Visualizing reconstruction...")
visualize_reconstruction(
    X_test[0], encoder, decoder,
    save_path="results/quick_example_reconstruction.png"
)
print("   ✓ Saved to: results/quick_example_reconstruction.png")

# 5. Summary
print("\n" + "="*70)
print("EXAMPLE COMPLETED!")
print("="*70)
print("\nWhat was demonstrated:")
print("  ✓ Data loading and preprocessing")
print("  ✓ PCA autoencoder architecture")
print("  ✓ Model training")
print("  ✓ Image reconstruction")
print("\nNext steps:")
print("  • Try the full CLI: python main.py --mode convolutional --epochs 25")
print("  • Train denoising autoencoder: python main.py --mode denoising")
print("  • Explore image retrieval: python main.py --mode retrieval")
print("  • Create morphing effects: python main.py --mode morphing")
print("\nSee README.md for full documentation.")
print("="*70 + "\n")


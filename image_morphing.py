"""Image morphing using autoencoder latent space interpolation.

This module implements image morphing by interpolating between encoded
representations in the latent space and decoding the intermediate points.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Optional
from utils.visualization import show_image
import os


def interpolate_images(
    image1: np.ndarray,
    image2: np.ndarray,
    encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    n_steps: int = 7
) -> np.ndarray:
    """
    Interpolate between two images in latent space.
    
    This function encodes two images, linearly interpolates between their
    latent codes, and decodes the intermediate representations to create
    a smooth morphing sequence.
    
    Args:
        image1: First image of shape (height, width, channels)
        image2: Second image of shape (height, width, channels)
        encoder: Trained encoder model
        decoder: Trained decoder model
        n_steps: Number of interpolation steps
    
    Returns:
        Array of interpolated images of shape (n_steps, height, width, channels)
    """
    # Encode both images
    if image1.ndim == 3:
        image1 = image1[np.newaxis, ...]
        image2 = image2[np.newaxis, ...]
    
    code1 = encoder.predict(image1, verbose=0)[0]
    code2 = encoder.predict(image2, verbose=0)[0]
    
    # Generate interpolation weights
    alphas = np.linspace(0, 1, n_steps)
    
    # Interpolate and decode
    interpolated_images = []
    for alpha in alphas:
        # Linear interpolation in latent space
        interpolated_code = code1 * (1 - alpha) + code2 * alpha
        
        # Decode
        decoded_image = decoder.predict(
            interpolated_code[np.newaxis, ...],
            verbose=0
        )[0]
        
        interpolated_images.append(decoded_image)
    
    return np.array(interpolated_images)


def visualize_morphing(
    image1: np.ndarray,
    image2: np.ndarray,
    encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    n_steps: int = 7,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize image morphing sequence.
    
    Args:
        image1: First image
        image2: Second image
        encoder: Trained encoder model
        decoder: Trained decoder model
        n_steps: Number of interpolation steps
        save_path: Optional path to save the figure
    """
    # Get interpolated images
    morphed_images = interpolate_images(
        image1, image2, encoder, decoder, n_steps
    )
    
    # Create figure
    fig, axes = plt.subplots(1, n_steps, figsize=(2.5 * n_steps, 3))
    
    if n_steps == 1:
        axes = [axes]
    
    # Display each step
    alphas = np.linspace(0, 1, n_steps)
    for i, (img, alpha) in enumerate(zip(morphed_images, alphas)):
        axes[i].set_title(f"α = {alpha:.2f}", fontsize=10)
        show_image(img, axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved morphing visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def demonstrate_morphing(
    encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    X_test: np.ndarray,
    n_pairs: int = 5,
    n_steps: int = 7,
    output_dir: str = 'results'
) -> None:
    """
    Demonstrate image morphing on random pairs of test images.
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model
        X_test: Test images
        n_pairs: Number of image pairs to morph
        n_steps: Number of interpolation steps
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("IMAGE MORPHING DEMONSTRATION")
    print(f"{'='*60}\n")
    
    for i in range(n_pairs):
        # Select random pair
        idx1, idx2 = np.random.choice(len(X_test), 2, replace=False)
        image1 = X_test[idx1]
        image2 = X_test[idx2]
        
        print(f"Morphing pair {i+1}/{n_pairs}...")
        
        save_path = os.path.join(output_dir, f'morphing_pair_{i+1}.png')
        visualize_morphing(
            image1, image2,
            encoder, decoder,
            n_steps=n_steps,
            save_path=save_path
        )
    
    print(f"\n{'='*60}")
    print(f"Morphing visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")


def create_morphing_grid(
    encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    X_test: np.ndarray,
    grid_size: int = 3,
    n_steps: int = 5,
    save_path: Optional[str] = None
) -> None:
    """
    Create a grid of morphing sequences between multiple images.
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model
        X_test: Test images
        grid_size: Number of rows/columns in the grid
        n_steps: Number of interpolation steps for each sequence
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(
        grid_size, n_steps,
        figsize=(2 * n_steps, 2 * grid_size)
    )
    
    for row in range(grid_size):
        # Select random pair for this row
        idx1, idx2 = np.random.choice(len(X_test), 2, replace=False)
        image1 = X_test[idx1]
        image2 = X_test[idx2]
        
        # Get morphing sequence
        morphed_images = interpolate_images(
            image1, image2, encoder, decoder, n_steps
        )
        
        # Display in row
        for col in range(n_steps):
            ax = axes[row, col] if grid_size > 1 else axes[col]
            show_image(morphed_images[col], ax)
            
            if row == 0:
                alpha = col / (n_steps - 1)
                ax.set_title(f"α={alpha:.1f}", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved morphing grid to {save_path}")
    else:
        plt.show()
    
    plt.close()


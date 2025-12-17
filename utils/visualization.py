"""Visualization utilities for autoencoders."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import tensorflow as tf


def show_image(image: np.ndarray, ax: Optional[plt.Axes] = None) -> None:
    """
    Display a single image.
    
    Args:
        image: Image array of shape (height, width, channels)
        ax: Matplotlib axes to plot on (creates new if None)
    """
    if ax is None:
        plt.imshow(np.clip(image + 0.5, 0, 1))
    else:
        ax.imshow(np.clip(image + 0.5, 0, 1))
        ax.axis('off')


def visualize_reconstruction(
    original: np.ndarray,
    encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize original image, its encoding, and reconstruction.
    
    Args:
        original: Original image of shape (height, width, channels)
        encoder: Encoder model
        decoder: Decoder model
        save_path: Optional path to save the figure
    """
    # Get code and reconstruction
    code = encoder.predict(original[np.newaxis, ...], verbose=0)[0]
    reconstruction = decoder.predict(code[np.newaxis, ...], verbose=0)[0]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    show_image(original, axes[0])
    
    # Code visualization
    axes[1].set_title("Latent Code", fontsize=12, fontweight='bold')
    code_2d = code.reshape([max(1, code.shape[-1] // 4), -1])
    axes[1].imshow(code_2d, cmap='viridis', aspect='auto')
    axes[1].axis('off')
    
    # Reconstruction
    axes[2].set_title("Reconstructed", fontsize=12, fontweight='bold')
    show_image(reconstruction, axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sample_images(
    images: np.ndarray,
    n_samples: int = 6,
    title: str = "Sample Images",
    save_path: Optional[str] = None
) -> None:
    """
    Plot a grid of sample images.
    
    Args:
        images: Array of images
        n_samples: Number of samples to display
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    n_samples = min(n_samples, len(images))
    rows = 2
    cols = (n_samples + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i in range(n_samples):
        show_image(images[i], axes[i])
    
    # Hide extra subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample images to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: tf.keras.callbacks.History,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss over epochs.
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot training loss
    ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
    
    # Plot validation loss if available
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_denoising(
    original: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize original, noisy, and denoised images side by side.
    
    Args:
        original: Original clean image
        noisy: Noisy input image
        denoised: Denoised output image
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    show_image(original, axes[0])
    
    axes[1].set_title("Noisy Input", fontsize=12, fontweight='bold')
    show_image(noisy, axes[1])
    
    axes[2].set_title("Denoised Output", fontsize=12, fontweight='bold')
    show_image(denoised, axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved denoising visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


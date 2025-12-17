"""Denoising autoencoder implementation.

This module implements a denoising autoencoder that learns to reconstruct
clean images from corrupted inputs. It uses the same architecture as the
convolutional autoencoder but trains on noisy inputs with clean targets.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple, Callable
import os

from .convolutional_autoencoder import build_convolutional_autoencoder
from utils.noise import apply_gaussian_noise


def build_denoising_autoencoder(
    input_shape: Tuple[int, int, int],
    code_size: int = 512,
    filters: Tuple[int, ...] = (32, 64, 128, 256)
) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """
    Build a denoising autoencoder.
    
    The architecture is the same as the convolutional autoencoder, but
    it's trained with noisy inputs to learn robust feature representations.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        code_size: Dimension of the latent code (larger for better quality)
        filters: Number of filters for each convolutional layer
    
    Returns:
        Tuple of (autoencoder, encoder, decoder) models
    
    Example:
        >>> model, enc, dec = build_denoising_autoencoder((32, 32, 3), code_size=512)
        >>> # Train with noisy inputs but clean targets
        >>> X_noisy = apply_gaussian_noise(X_train, sigma=0.1)
        >>> model.fit(X_noisy, X_train, epochs=25)
    """
    # Use the same architecture as convolutional autoencoder
    # but with larger code size for better denoising quality
    return build_convolutional_autoencoder(
        input_shape=input_shape,
        code_size=code_size,
        filters=filters
    )


def train_denoising_autoencoder(
    autoencoder: keras.Model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    noise_function: Callable = apply_gaussian_noise,
    noise_params: dict = None,
    epochs: int = 25,
    batch_size: int = 32,
    checkpoint_dir: str = 'checkpoints',
    verbose: int = 1
) -> keras.callbacks.History:
    """
    Train the denoising autoencoder with data augmentation.
    
    This function trains the autoencoder by corrupting inputs with noise
    at each epoch, forcing the model to learn robust denoising features.
    
    Args:
        autoencoder: The autoencoder model to train
        X_train: Training images (clean)
        X_test: Test images (clean)
        noise_function: Function to apply noise (default: Gaussian)
        noise_params: Parameters for noise function (e.g., {'sigma': 0.1})
        epochs: Number of training epochs
        batch_size: Batch size for training
        checkpoint_dir: Directory to save model checkpoints
        verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
    
    Returns:
        Training history object
    """
    if noise_params is None:
        noise_params = {'sigma': 0.1}
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Compile model
    autoencoder.compile(
        optimizer='adamax',
        loss='mse',
        metrics=['mae']
    )
    
    print("\nTraining Denoising Autoencoder...")
    print(f"Noise parameters: {noise_params}")
    print(f"This may take a while depending on your hardware.")
    
    # Custom training loop with noise augmentation
    history = {
        'loss': [],
        'val_loss': [],
        'mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("Generating corrupted training samples...")
        
        # Apply noise to training data
        X_train_noisy = noise_function(X_train, **noise_params)
        X_test_noisy = noise_function(X_test, **noise_params)
        
        # Train for one epoch
        epoch_history = autoencoder.fit(
            X_train_noisy, X_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(X_test_noisy, X_test),
            verbose=verbose
        )
        
        # Store history
        history['loss'].append(epoch_history.history['loss'][0])
        history['val_loss'].append(epoch_history.history['val_loss'][0])
        history['mae'].append(epoch_history.history['mae'][0])
        history['val_mae'].append(epoch_history.history['val_mae'][0])
        
        # Check for improvement
        current_val_loss = epoch_history.history['val_loss'][0]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Save best model
            autoencoder.save(
                os.path.join(checkpoint_dir, 'denoising_autoencoder_best.keras')
            )
            print(f"Validation loss improved to {best_val_loss:.6f}, saving model...")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    # Load best model
    autoencoder = keras.models.load_model(
        os.path.join(checkpoint_dir, 'denoising_autoencoder_best.keras')
    )
    
    # Evaluate on noisy test data
    X_test_noisy = noise_function(X_test, **noise_params)
    test_loss = autoencoder.evaluate(X_test_noisy, X_test, verbose=0)
    print(f"\nDenoising Autoencoder - Test MSE: {test_loss[0]:.6f}")
    
    # Create history object
    class HistoryObject:
        def __init__(self, history_dict):
            self.history = history_dict
    
    return HistoryObject(history)


def denoise_images(
    autoencoder: keras.Model,
    noisy_images: np.ndarray,
    batch_size: int = 32
) -> np.ndarray:
    """
    Denoise images using the trained autoencoder.
    
    Args:
        autoencoder: Trained denoising autoencoder
        noisy_images: Noisy input images
        batch_size: Batch size for prediction
    
    Returns:
        Denoised images
    """
    denoised = autoencoder.predict(noisy_images, batch_size=batch_size, verbose=0)
    return denoised


"""Linear PCA-style autoencoder implementation.

This module implements a simple linear autoencoder that performs
dimensionality reduction similar to Principal Component Analysis (PCA).
The autoencoder consists of fully connected layers without non-linear
activations.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple


def build_pca_autoencoder(
    input_shape: Tuple[int, int, int],
    code_size: int = 32
) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """
    Build a linear autoencoder similar to PCA.
    
    This autoencoder uses only linear transformations (Dense layers without
    activation functions) to compress and reconstruct images. It's mathematically
    similar to PCA but learned through backpropagation.
    
    Architecture:
        Encoder: Input -> Flatten -> Dense(code_size)
        Decoder: Dense(input_dims) -> Reshape -> Output
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        code_size: Dimension of the latent code (bottleneck size)
    
    Returns:
        Tuple of (autoencoder, encoder, decoder) models
    
    Example:
        >>> autoencoder, encoder, decoder = build_pca_autoencoder((32, 32, 3), code_size=64)
        >>> autoencoder.compile(optimizer='adam', loss='mse')
        >>> autoencoder.fit(X_train, X_train, epochs=10)
    """
    # Calculate total input dimensions
    input_dims = np.prod(input_shape)
    
    # ==================== ENCODER ====================
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Flatten(name='flatten')(encoder_input)
    encoder_output = layers.Dense(code_size, name='latent_code')(x)
    
    encoder = keras.Model(
        inputs=encoder_input,
        outputs=encoder_output,
        name='pca_encoder'
    )
    
    # ==================== DECODER ====================
    decoder_input = layers.Input(shape=(code_size,), name='decoder_input')
    x = layers.Dense(input_dims, name='decoder_dense')(decoder_input)
    decoder_output = layers.Reshape(input_shape, name='reshape')(x)
    
    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output,
        name='pca_decoder'
    )
    
    # ==================== AUTOENCODER ====================
    autoencoder_input = layers.Input(shape=input_shape, name='input')
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    
    autoencoder = keras.Model(
        inputs=autoencoder_input,
        outputs=decoded,
        name='pca_autoencoder'
    )
    
    return autoencoder, encoder, decoder


def train_pca_autoencoder(
    autoencoder: keras.Model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    verbose: int = 1
) -> keras.callbacks.History:
    """
    Train the PCA autoencoder.
    
    Args:
        autoencoder: The autoencoder model to train
        X_train: Training images
        X_test: Test images for validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
    
    Returns:
        Training history object
    """
    # Compile model
    autoencoder.compile(
        optimizer='adamax',
        loss='mse',
        metrics=['mae']
    )
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=verbose
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=verbose
        )
    ]
    
    # Train model
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, X_test),
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Evaluate
    test_loss = autoencoder.evaluate(X_test, X_test, verbose=0)
    print(f"\nPCA Autoencoder - Test MSE: {test_loss[0]:.6f}")
    
    return history


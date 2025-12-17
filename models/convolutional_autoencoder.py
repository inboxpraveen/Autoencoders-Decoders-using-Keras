"""Deep convolutional autoencoder implementation.

This module implements a deep convolutional autoencoder that uses
convolutional layers for encoding and transpose convolutions for decoding.
This architecture is much more powerful than linear PCA for image data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional
import os


def build_convolutional_autoencoder(
    input_shape: Tuple[int, int, int],
    code_size: int = 32,
    filters: Tuple[int, ...] = (32, 64, 128, 256)
) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """
    Build a deep convolutional autoencoder.
    
    The encoder uses convolutional layers with max pooling to progressively
    reduce spatial dimensions while increasing feature depth. The decoder
    uses transpose convolutions to reconstruct the original image.
    
    Architecture:
        Encoder: Conv2D + MaxPool (repeated) -> Flatten -> Dense(code_size)
        Decoder: Dense -> Reshape -> Conv2DTranspose (repeated)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        code_size: Dimension of the latent code (bottleneck size)
        filters: Number of filters for each convolutional layer
    
    Returns:
        Tuple of (autoencoder, encoder, decoder) models
    
    Example:
        >>> model, enc, dec = build_convolutional_autoencoder((32, 32, 3), code_size=32)
        >>> model.compile(optimizer='adam', loss='mse')
        >>> model.fit(X_train, X_train, epochs=25)
    """
    height, width, channels = input_shape
    
    # ==================== ENCODER ====================
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    x = encoder_input
    
    # Stack convolutional layers with max pooling
    for i, n_filters in enumerate(filters):
        x = layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name=f'encoder_conv_{i+1}'
        )(x)
        x = layers.MaxPooling2D(
            pool_size=(2, 2),
            name=f'encoder_pool_{i+1}'
        )(x)
    
    # Get shape before flattening (needed for decoder)
    shape_before_flatten = x.shape[1:]
    
    # Flatten and create bottleneck
    x = layers.Flatten(name='flatten')(x)
    encoder_output = layers.Dense(
        code_size,
        activation='relu',
        name='latent_code'
    )(x)
    
    encoder = keras.Model(
        inputs=encoder_input,
        outputs=encoder_output,
        name='convolutional_encoder'
    )
    
    # ==================== DECODER ====================
    decoder_input = layers.Input(shape=(code_size,), name='decoder_input')
    
    # Dense layer to reshape
    decoder_dense_units = np.prod(shape_before_flatten)
    x = layers.Dense(
        decoder_dense_units,
        activation='relu',
        name='decoder_dense'
    )(decoder_input)
    x = layers.Reshape(shape_before_flatten, name='reshape')(x)
    
    # Stack transpose convolutional layers
    for i, n_filters in enumerate(reversed(filters[:-1])):
        x = layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu',
            name=f'decoder_deconv_{i+1}'
        )(x)
    
    # Final layer to reconstruct image
    decoder_output = layers.Conv2DTranspose(
        filters=channels,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        activation=None,
        name='output'
    )(x)
    
    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output,
        name='convolutional_decoder'
    )
    
    # ==================== AUTOENCODER ====================
    autoencoder_input = layers.Input(shape=input_shape, name='input')
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    
    autoencoder = keras.Model(
        inputs=autoencoder_input,
        outputs=decoded,
        name='convolutional_autoencoder'
    )
    
    return autoencoder, encoder, decoder


def train_convolutional_autoencoder(
    autoencoder: keras.Model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 25,
    batch_size: int = 32,
    checkpoint_dir: str = 'checkpoints',
    verbose: int = 1
) -> keras.callbacks.History:
    """
    Train the convolutional autoencoder.
    
    Args:
        autoencoder: The autoencoder model to train
        X_train: Training images
        X_test: Test images for validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        checkpoint_dir: Directory to save model checkpoints
        verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
    
    Returns:
        Training history object
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Compile model
    autoencoder.compile(
        optimizer='adamax',
        loss='mse',
        metrics=['mae']
    )
    
    # Define callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'conv_autoencoder_epoch_{epoch:03d}.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=verbose
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=verbose
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', 'convolutional'),
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nTraining Convolutional Autoencoder...")
    print(f"This may take a while depending on your hardware.")
    
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
    print(f"\nConvolutional Autoencoder - Test MSE: {test_loss[0]:.6f}")
    
    return history


def save_weights(
    encoder: keras.Model,
    decoder: keras.Model,
    save_dir: str = 'saved_models'
) -> None:
    """
    Save encoder and decoder weights separately.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        save_dir: Directory to save weights
    """
    os.makedirs(save_dir, exist_ok=True)
    
    encoder_path = os.path.join(save_dir, 'encoder.weights.h5')
    decoder_path = os.path.join(save_dir, 'decoder.weights.h5')
    
    encoder.save_weights(encoder_path)
    decoder.save_weights(decoder_path)
    
    print(f"\nSaved encoder weights to: {encoder_path}")
    print(f"Saved decoder weights to: {decoder_path}")


def load_weights(
    encoder: keras.Model,
    decoder: keras.Model,
    save_dir: str = 'saved_models'
) -> None:
    """
    Load encoder and decoder weights.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        save_dir: Directory containing saved weights
    """
    encoder_path = os.path.join(save_dir, 'encoder.weights.h5')
    decoder_path = os.path.join(save_dir, 'decoder.weights.h5')
    
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        encoder.load_weights(encoder_path)
        decoder.load_weights(decoder_path)
        print(f"\nLoaded encoder weights from: {encoder_path}")
        print(f"Loaded decoder weights from: {decoder_path}")
    else:
        raise FileNotFoundError(f"Weights not found in {save_dir}")


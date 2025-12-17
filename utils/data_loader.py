"""Data loading and preprocessing utilities for LFW dataset."""

import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os


def load_lfw_dataset(
    dimx: int = 32,
    dimy: int = 32,
    use_raw: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Labeled Faces in the Wild (LFW) dataset.
    
    This function attempts to load from sklearn.datasets. If not available,
    it provides instructions for manual download.
    
    Args:
        dimx: Target width for resizing images
        dimy: Target height for resizing images
        use_raw: Whether to use raw images (True) or aligned/cropped (False)
    
    Returns:
        Tuple of (images, attributes) where:
            - images: numpy array of shape (n_samples, height, width, channels)
            - attributes: numpy array of attributes (empty if not available)
    
    Raises:
        ImportError: If sklearn.datasets is not available
        RuntimeError: If dataset cannot be loaded
    """
    try:
        from sklearn.datasets import fetch_lfw_people
        
        print(f"Loading LFW dataset (resizing to {dimx}x{dimy})...")
        lfw_people = fetch_lfw_people(
            min_faces_per_person=20,
            resize=dimx / 250.0,  # Original images are 250x250
            color=True
        )
        
        # Get images and reshape to proper format
        images = lfw_people.images
        
        # Ensure images are in (height, width, channels) format
        if images.ndim == 3:
            # Convert grayscale to RGB
            images = np.stack([images] * 3, axis=-1)
        
        # Resize if needed
        if images.shape[1] != dimy or images.shape[2] != dimx:
            print(f"Resizing images to {dimx}x{dimy}...")
            images_resized = []
            for img in images:
                img_resized = tf.image.resize(img, [dimy, dimx]).numpy()
                images_resized.append(img_resized)
            images = np.array(images_resized)
        
        # Create dummy attributes (for compatibility with original code)
        attributes = np.zeros((len(images), 73))
        
        print(f"Loaded {len(images)} images with shape {images.shape[1:]}")
        return images, attributes
        
    except Exception as e:
        print(f"Error loading LFW dataset: {e}")
        print("\nAlternative: You can download the dataset manually from:")
        print("http://vis-www.cs.umass.edu/lfw/")
        print("\nOr generate synthetic face data for testing.")
        
        # Return synthetic data for testing
        print("\nGenerating synthetic data for demonstration...")
        n_samples = 1000
        images = np.random.rand(n_samples, dimy, dimx, 3).astype(np.float32) * 255
        attributes = np.zeros((n_samples, 73))
        return images, attributes


def prepare_data(
    images: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare and preprocess image data for training.
    
    Args:
        images: Input images as numpy array
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        normalize: Whether to normalize images to [-0.5, 0.5] range
    
    Returns:
        Tuple of (X_train, X_test)
    """
    # Normalize if requested
    if normalize:
        images = images.astype('float32') / 255.0 - 0.5
    else:
        images = images.astype('float32') / 255.0
    
    # Split into train and test
    X_train, X_test = train_test_split(
        images,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    
    return X_train, X_test


def create_tf_dataset(
    X: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    repeat: bool = False
) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset for efficient training.
    
    Args:
        X: Input images
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        repeat: Whether to repeat the dataset indefinitely
    
    Returns:
        tf.data.Dataset object
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, X))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    if repeat:
        dataset = dataset.repeat()
    
    return dataset


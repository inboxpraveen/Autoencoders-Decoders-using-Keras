"""Data loading and preprocessing utilities for LFW and custom image folders."""

import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')


def load_lfw_dataset(
    dimx: int = 32,
    dimy: int = 32,
    min_faces_per_person: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Labeled Faces in the Wild (LFW) dataset.

    Downloads the dataset through scikit-learn on first use (about 200 MB,
    cached in ~/scikit_learn_data). A square 125x125 crop around the face is
    taken from each 250x250 photo and resized to the target size. If the
    download fails, random images are returned so the pipeline can still be
    exercised, and a warning is printed.

    Args:
        dimx: Target width for resizing images
        dimy: Target height for resizing images
        min_faces_per_person: Keep only people with at least this many
            photos (0 keeps all 13,233 images)

    Returns:
        Tuple of (images, attributes) where:
            - images: float32 array of shape (n_samples, height, width, 3)
              with pixel values in [0, 255]
            - attributes: placeholder array of zeros, kept for compatibility
    """
    try:
        from sklearn.datasets import fetch_lfw_people

        print(f"Loading LFW dataset (resizing to {dimx}x{dimy})...")
        lfw_people = fetch_lfw_people(
            min_faces_per_person=min_faces_per_person,
            slice_=(slice(70, 195), slice(62, 187)),  # 125x125 face crop
            resize=dimx / 125.0,
            color=True
        )

        # scikit-learn returns pixel values in [0, 1]; scale to [0, 255]
        # so every loader hands prepare_data() the same range
        images = lfw_people.images * 255.0

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
        print("\nAlternative: download the dataset manually from")
        print("http://vis-www.cs.umass.edu/lfw/ or train on your own images")
        print("with --data-dir path/to/images")

        # Return random data so the rest of the pipeline can still run
        print("\nWARNING: falling back to random images. Results will be meaningless.")
        n_samples = 1000
        images = np.random.rand(n_samples, dimy, dimx, 3).astype(np.float32) * 255
        attributes = np.zeros((n_samples, 73))
        return images, attributes


def load_image_folder(
    data_dir: str,
    dimx: int = 32,
    dimy: int = 32
) -> np.ndarray:
    """
    Load every image in a folder (subfolders included) as RGB arrays.

    Use this to train on your own data instead of LFW. Images are converted
    to RGB and resized to dimx x dimy without keeping the aspect ratio, so
    center-cropped or roughly square images give the best results.

    Args:
        data_dir: Folder containing .jpg, .jpeg, .png, .bmp, .gif or .webp files
        dimx: Target width
        dimy: Target height

    Returns:
        float32 array of shape (n_samples, dimy, dimx, 3) with values in [0, 255]

    Raises:
        FileNotFoundError: If the folder does not exist or holds no images
    """
    from PIL import Image

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Image folder not found: {data_dir}")

    paths = []
    for root, _, files in os.walk(data_dir):
        for name in sorted(files):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, name))

    if not paths:
        raise FileNotFoundError(f"No images found in: {data_dir}")

    print(f"Loading {len(paths)} images from {data_dir} (resizing to {dimx}x{dimy})...")
    images = np.empty((len(paths), dimy, dimx, 3), dtype=np.float32)
    for i, path in enumerate(sorted(paths)):
        with Image.open(path) as img:
            img = img.convert('RGB').resize((dimx, dimy), Image.BILINEAR)
            images[i] = np.asarray(img, dtype=np.float32)

    print(f"Loaded {len(images)} images with shape {images.shape[1:]}")
    return images


def prepare_data(
    images: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare and preprocess image data for training.
    
    Args:
        images: Input images as numpy array with pixel values in [0, 255]
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


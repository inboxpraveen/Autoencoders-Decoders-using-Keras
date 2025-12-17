"""Noise utilities for denoising autoencoders."""

import numpy as np
from typing import Optional


def apply_gaussian_noise(
    images: np.ndarray,
    sigma: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to images.
    
    This function adds zero-mean Gaussian noise with specified standard
    deviation to corrupt the input images. Useful for training denoising
    autoencoders.
    
    Args:
        images: Input images of shape (n_samples, height, width, channels)
        sigma: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        Noisy images with the same shape as input
    
    Example:
        >>> X_noisy = apply_gaussian_noise(X_train, sigma=0.1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(loc=0.0, scale=sigma, size=images.shape)
    noisy_images = images + noise
    
    return noisy_images.astype(images.dtype)


def apply_salt_pepper_noise(
    images: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add salt and pepper noise to images.
    
    Args:
        images: Input images of shape (n_samples, height, width, channels)
        amount: Proportion of pixels to corrupt
        salt_vs_pepper: Proportion of salt vs pepper (0.5 means equal)
        seed: Random seed for reproducibility
    
    Returns:
        Noisy images with the same shape as input
    """
    if seed is not None:
        np.random.seed(seed)
    
    noisy_images = images.copy()
    
    # Generate random values
    random_values = np.random.rand(*images.shape)
    
    # Add salt (white pixels)
    salt_mask = random_values < (amount * salt_vs_pepper)
    noisy_images[salt_mask] = 0.5  # Maximum value in normalized range
    
    # Add pepper (black pixels)
    pepper_mask = random_values > (1 - amount * (1 - salt_vs_pepper))
    noisy_images[pepper_mask] = -0.5  # Minimum value in normalized range
    
    return noisy_images


def apply_random_occlusion(
    images: np.ndarray,
    n_occlusions: int = 3,
    occlusion_size: int = 8,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add random black rectangular occlusions to images.
    
    Args:
        images: Input images of shape (n_samples, height, width, channels)
        n_occlusions: Number of occlusions per image
        occlusion_size: Size of each square occlusion
        seed: Random seed for reproducibility
    
    Returns:
        Images with random occlusions
    """
    if seed is not None:
        np.random.seed(seed)
    
    noisy_images = images.copy()
    n_samples, height, width, channels = images.shape
    
    for i in range(n_samples):
        for _ in range(n_occlusions):
            # Random position
            y = np.random.randint(0, height - occlusion_size + 1)
            x = np.random.randint(0, width - occlusion_size + 1)
            
            # Apply occlusion
            noisy_images[i, y:y+occlusion_size, x:x+occlusion_size, :] = -0.5
    
    return noisy_images


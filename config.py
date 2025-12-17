"""Configuration settings for the autoencoder project.

This module contains default configurations that can be imported and
customized for different experiments.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    image_size: int = 32
    test_size: float = 0.1
    random_state: int = 42
    normalize: bool = True


@dataclass
class PCAConfig:
    """PCA autoencoder configuration."""
    code_size: int = 32
    epochs: int = 15
    batch_size: int = 32
    optimizer: str = 'adamax'
    loss: str = 'mse'


@dataclass
class ConvolutionalConfig:
    """Convolutional autoencoder configuration."""
    code_size: int = 32
    filters: Tuple[int, ...] = (32, 64, 128, 256)
    epochs: int = 25
    batch_size: int = 32
    optimizer: str = 'adamax'
    loss: str = 'mse'


@dataclass
class DenoisingConfig:
    """Denoising autoencoder configuration."""
    code_size: int = 512
    filters: Tuple[int, ...] = (32, 64, 128, 256)
    epochs: int = 25
    batch_size: int = 32
    noise_sigma: float = 0.1
    optimizer: str = 'adamax'
    loss: str = 'mse'


@dataclass
class PathConfig:
    """Path configuration for saving/loading models."""
    save_dir: str = 'saved_models'
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    logs_dir: str = 'logs'


# Default configurations
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_PCA_CONFIG = PCAConfig()
DEFAULT_CONV_CONFIG = ConvolutionalConfig()
DEFAULT_DENOISING_CONFIG = DenoisingConfig()
DEFAULT_PATH_CONFIG = PathConfig()


"""Main entry point for Autoencoder Training and Demonstration.

This script provides a comprehensive command-line interface for training
and using various autoencoder architectures on image data.

Usage:
    python main.py --mode pca --epochs 15
    python main.py --mode convolutional --epochs 25 --code-size 32
    python main.py --mode denoising --epochs 25 --code-size 512 --noise-sigma 0.1
    python main.py --mode retrieval --model-path saved_models
    python main.py --mode morphing --model-path saved_models
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from typing import Optional

# Import project modules
from utils import (
    load_lfw_dataset,
    prepare_data,
    plot_sample_images,
    plot_training_history,
    visualize_reconstruction,
    visualize_denoising,
    apply_gaussian_noise
)
from models import (
    build_pca_autoencoder,
    build_convolutional_autoencoder,
    build_denoising_autoencoder
)
from models.pca_autoencoder import train_pca_autoencoder
from models.convolutional_autoencoder import (
    train_convolutional_autoencoder,
    save_weights,
    load_weights
)
from models.denoising_autoencoder import train_denoising_autoencoder
from image_retrieval import demonstrate_retrieval
from image_morphing import demonstrate_morphing

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def setup_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("✓ No GPU found, using CPU")


def train_pca_mode(args, X_train, X_test, img_shape):
    """Train PCA-style linear autoencoder."""
    print("\n" + "="*70)
    print("TRAINING PCA AUTOENCODER")
    print("="*70)
    
    # Build model
    autoencoder, encoder, decoder = build_pca_autoencoder(
        input_shape=img_shape,
        code_size=args.code_size
    )
    
    print(f"\n📊 Model Configuration:")
    print(f"   - Code size: {args.code_size}")
    print(f"   - Input shape: {img_shape}")
    print(f"   - Total parameters: {autoencoder.count_params():,}")
    
    # Train
    history = train_pca_autoencoder(
        autoencoder, X_train, X_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # Visualize results
    output_dir = 'results/pca'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plot_training_history(history, save_path=f"{output_dir}/training_history.png")
    
    # Visualize reconstructions
    for i in range(min(5, len(X_test))):
        visualize_reconstruction(
            X_test[i], encoder, decoder,
            save_path=f"{output_dir}/reconstruction_{i+1}.png"
        )
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return autoencoder, encoder, decoder


def train_convolutional_mode(args, X_train, X_test, img_shape):
    """Train deep convolutional autoencoder."""
    print("\n" + "="*70)
    print("TRAINING CONVOLUTIONAL AUTOENCODER")
    print("="*70)
    
    # Build model
    autoencoder, encoder, decoder = build_convolutional_autoencoder(
        input_shape=img_shape,
        code_size=args.code_size
    )
    
    print(f"\n📊 Model Configuration:")
    print(f"   - Code size: {args.code_size}")
    print(f"   - Input shape: {img_shape}")
    print(f"   - Total parameters: {autoencoder.count_params():,}")
    
    print("\n🔧 Encoder Architecture:")
    encoder.summary()
    
    print("\n🔧 Decoder Architecture:")
    decoder.summary()
    
    # Train
    history = train_convolutional_autoencoder(
        autoencoder, X_train, X_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        verbose=1
    )
    
    # Save weights
    save_weights(encoder, decoder, save_dir=args.save_dir)
    
    # Visualize results
    output_dir = 'results/convolutional'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_training_history(history, save_path=f"{output_dir}/training_history.png")
    
    for i in range(min(5, len(X_test))):
        visualize_reconstruction(
            X_test[i], encoder, decoder,
            save_path=f"{output_dir}/reconstruction_{i+1}.png"
        )
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return autoencoder, encoder, decoder


def train_denoising_mode(args, X_train, X_test, img_shape):
    """Train denoising autoencoder."""
    print("\n" + "="*70)
    print("TRAINING DENOISING AUTOENCODER")
    print("="*70)
    
    # Build model
    autoencoder, encoder, decoder = build_denoising_autoencoder(
        input_shape=img_shape,
        code_size=args.code_size
    )
    
    print(f"\n📊 Model Configuration:")
    print(f"   - Code size: {args.code_size}")
    print(f"   - Input shape: {img_shape}")
    print(f"   - Noise sigma: {args.noise_sigma}")
    print(f"   - Total parameters: {autoencoder.count_params():,}")
    
    # Train
    history = train_denoising_autoencoder(
        autoencoder, X_train, X_test,
        noise_function=apply_gaussian_noise,
        noise_params={'sigma': args.noise_sigma},
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        verbose=1
    )
    
    # Save weights
    save_weights(encoder, decoder, save_dir=args.save_dir)
    
    # Visualize results
    output_dir = 'results/denoising'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_training_history(history, save_path=f"{output_dir}/training_history.png")
    
    # Visualize denoising
    for i in range(min(5, len(X_test))):
        original = X_test[i]
        noisy = apply_gaussian_noise(
            X_test[i:i+1],
            sigma=args.noise_sigma
        )[0]
        denoised = autoencoder.predict(noisy[np.newaxis, ...], verbose=0)[0]
        
        visualize_denoising(
            original, noisy, denoised,
            save_path=f"{output_dir}/denoising_{i+1}.png"
        )
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return autoencoder, encoder, decoder


def retrieval_mode(args, X_train, X_test, img_shape):
    """Run image retrieval demonstration."""
    print("\n" + "="*70)
    print("IMAGE RETRIEVAL MODE")
    print("="*70)
    
    # Build and load model
    _, encoder, _ = build_convolutional_autoencoder(
        input_shape=img_shape,
        code_size=args.code_size
    )
    
    try:
        encoder.load_weights(os.path.join(args.model_path, 'encoder.weights.h5'))
        print(f"✓ Loaded encoder from: {args.model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train a model first using --mode convolutional")
        return
    
    # Run retrieval demonstration
    demonstrate_retrieval(
        encoder, X_train, X_test,
        n_queries=args.n_queries,
        n_neighbors=args.n_neighbors,
        output_dir='results/retrieval'
    )


def morphing_mode(args, X_train, X_test, img_shape):
    """Run image morphing demonstration."""
    print("\n" + "="*70)
    print("IMAGE MORPHING MODE")
    print("="*70)
    
    # Build and load model
    _, encoder, decoder = build_convolutional_autoencoder(
        input_shape=img_shape,
        code_size=args.code_size
    )
    
    try:
        encoder.load_weights(os.path.join(args.model_path, 'encoder.weights.h5'))
        decoder.load_weights(os.path.join(args.model_path, 'decoder.weights.h5'))
        print(f"✓ Loaded models from: {args.model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train a model first using --mode convolutional")
        return
    
    # Run morphing demonstration
    demonstrate_morphing(
        encoder, decoder, X_test,
        n_pairs=args.n_pairs,
        n_steps=args.n_steps,
        output_dir='results/morphing'
    )


def main():
    """Main function to parse arguments and run selected mode."""
    parser = argparse.ArgumentParser(
        description="Autoencoder Training and Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train PCA autoencoder:
    python main.py --mode pca --epochs 15 --code-size 32
  
  Train convolutional autoencoder:
    python main.py --mode convolutional --epochs 25 --code-size 32
  
  Train denoising autoencoder:
    python main.py --mode denoising --epochs 25 --code-size 512 --noise-sigma 0.1
  
  Run image retrieval:
    python main.py --mode retrieval --model-path saved_models
  
  Run image morphing:
    python main.py --mode morphing --model-path saved_models --n-pairs 5
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['pca', 'convolutional', 'denoising', 'retrieval', 'morphing'],
        help='Operation mode: pca, convolutional, denoising, retrieval, or morphing'
    )
    
    # Data parameters
    parser.add_argument(
        '--image-size',
        type=int,
        default=32,
        help='Image size for loading (default: 32)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.1,
        help='Test set fraction (default: 0.1)'
    )
    
    # Model parameters
    parser.add_argument(
        '--code-size',
        type=int,
        default=32,
        help='Latent code dimension (default: 32, use 512 for denoising)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of training epochs (default: 25)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    # Denoising parameters
    parser.add_argument(
        '--noise-sigma',
        type=float,
        default=0.1,
        help='Gaussian noise standard deviation for denoising (default: 0.1)'
    )
    
    # Retrieval parameters
    parser.add_argument(
        '--n-queries',
        type=int,
        default=3,
        help='Number of query images for retrieval (default: 3)'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=5,
        help='Number of similar images to retrieve (default: 5)'
    )
    
    # Morphing parameters
    parser.add_argument(
        '--n-pairs',
        type=int,
        default=5,
        help='Number of image pairs for morphing (default: 5)'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=7,
        help='Number of interpolation steps (default: 7)'
    )
    
    # Paths
    parser.add_argument(
        '--save-dir',
        type=str,
        default='saved_models',
        help='Directory to save trained models (default: saved_models)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for training checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='saved_models',
        help='Path to load trained models (default: saved_models)'
    )
    
    args = parser.parse_args()
    
    # Setup
    print("\n" + "="*70)
    print("AUTOENCODER PROJECT - Image Reconstruction & Generation")
    print("="*70)
    
    setup_gpu()
    
    # Load and prepare data
    print(f"\n📁 Loading dataset...")
    X, attr = load_lfw_dataset(dimx=args.image_size, dimy=args.image_size)
    X_train, X_test = prepare_data(X, test_size=args.test_size)
    img_shape = X_train.shape[1:]
    
    # Show sample images
    print(f"\n🖼️  Sample images from dataset:")
    plot_sample_images(X_train, n_samples=6, save_path='results/sample_images.png')
    
    # Execute selected mode
    mode_functions = {
        'pca': train_pca_mode,
        'convolutional': train_convolutional_mode,
        'denoising': train_denoising_mode,
        'retrieval': retrieval_mode,
        'morphing': morphing_mode
    }
    
    mode_functions[args.mode](args, X_train, X_test, img_shape)
    
    print("\n" + "="*70)
    print("✓ COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


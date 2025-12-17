"""Image retrieval using autoencoder latent representations.

This module implements similarity-based image search using the learned
latent space of autoencoders. Images are encoded into a low-dimensional
representation, and similar images are found using nearest neighbors search.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from typing import Tuple, Optional
from utils.visualization import show_image


class ImageRetrieval:
    """
    Image retrieval system using autoencoder embeddings.
    
    This class encodes a database of images using a trained encoder,
    then enables fast similarity search using k-nearest neighbors.
    """
    
    def __init__(
        self,
        encoder: tf.keras.Model,
        metric: str = 'euclidean'
    ):
        """
        Initialize the image retrieval system.
        
        Args:
            encoder: Trained encoder model
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        """
        self.encoder = encoder
        self.metric = metric
        self.neighbor_model = None
        self.image_database = None
        self.codes = None
    
    def index_images(self, images: np.ndarray) -> None:
        """
        Index a database of images for retrieval.
        
        Args:
            images: Array of images to index
        """
        print(f"Encoding {len(images)} images...")
        self.image_database = images
        self.codes = self.encoder.predict(images, verbose=0)
        
        print(f"Building nearest neighbor index with {self.metric} metric...")
        self.neighbor_model = NearestNeighbors(
            metric=self.metric,
            algorithm='auto'
        )
        self.neighbor_model.fit(self.codes)
        
        print(f"Indexed {len(images)} images successfully!")
    
    def find_similar(
        self,
        query_image: np.ndarray,
        n_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar images to the query image.
        
        Args:
            query_image: Query image of shape (height, width, channels)
            n_neighbors: Number of similar images to retrieve
        
        Returns:
            Tuple of (distances, similar_images)
        """
        if self.neighbor_model is None:
            raise ValueError("No images indexed. Call index_images() first.")
        
        # Ensure query is 4D
        if query_image.ndim == 3:
            query_image = query_image[np.newaxis, ...]
        
        # Encode query
        query_code = self.encoder.predict(query_image, verbose=0)
        
        # Find nearest neighbors
        distances, indices = self.neighbor_model.kneighbors(
            query_code,
            n_neighbors=n_neighbors
        )
        
        # Get similar images
        similar_images = self.image_database[indices[0]]
        
        return distances[0], similar_images
    
    def visualize_similar(
        self,
        query_image: np.ndarray,
        n_neighbors: int = 5,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize query image and its most similar images.
        
        Args:
            query_image: Query image
            n_neighbors: Number of similar images to show
            save_path: Optional path to save the figure
        """
        distances, similar_images = self.find_similar(query_image, n_neighbors)
        
        # Create figure
        fig, axes = plt.subplots(1, n_neighbors + 1, figsize=(3 * (n_neighbors + 1), 3))
        
        # Show query image
        axes[0].set_title("Query Image", fontsize=12, fontweight='bold')
        show_image(query_image, axes[0])
        
        # Show similar images
        for i in range(n_neighbors):
            axes[i + 1].set_title(
                f"#{i+1}\nDist: {distances[i]:.3f}",
                fontsize=10
            )
            show_image(similar_images[i], axes[i + 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def demonstrate_retrieval(
    encoder: tf.keras.Model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_queries: int = 3,
    n_neighbors: int = 5,
    output_dir: str = 'results'
) -> None:
    """
    Demonstrate image retrieval on test images.
    
    Args:
        encoder: Trained encoder model
        X_train: Training images (database)
        X_test: Test images (queries)
        n_queries: Number of query images to test
        n_neighbors: Number of similar images to retrieve
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create retrieval system
    retrieval = ImageRetrieval(encoder)
    retrieval.index_images(X_train)
    
    print(f"\n{'='*60}")
    print("IMAGE RETRIEVAL DEMONSTRATION")
    print(f"{'='*60}\n")
    
    # Test on random query images
    query_indices = np.random.choice(len(X_test), n_queries, replace=False)
    
    for i, idx in enumerate(query_indices):
        print(f"\nQuery {i+1}/{n_queries}:")
        query_image = X_test[idx]
        
        save_path = os.path.join(output_dir, f'retrieval_query_{i+1}.png')
        retrieval.visualize_similar(
            query_image,
            n_neighbors=n_neighbors,
            save_path=save_path
        )
    
    print(f"\n{'='*60}")
    print(f"Retrieval visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")


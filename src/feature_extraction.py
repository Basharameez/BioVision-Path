import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class FeatureExtractor:
    """
    Utility class to register a forward hook on a specific layer of a model
    to extract activations during inference.
    """
    def __init__(self, model, layer_name="avgpool"):
        self.model = model
        self.features = []
        # Find the target module by name
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise ValueError(f"Layer '{layer_name}' not found in model modules. Available: {list(modules.keys())}")
        self.target_layer = modules[layer_name]
        self.hook = self.target_layer.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        # For ResNet-18 avgpool, output is (batch_size, 512, 1, 1)
        self.features.append(output.detach().cpu())
        
    def clear(self):
        self.features = []
        
    def close(self):
        self.hook.remove()


@torch.no_grad()
def extract_resnet18_embeddings(model, loader, device, layer_name="avgpool"):
    """
    Extracts embeddings (features) from the penultimate layer of the ResNet-18 model.
    Returns:
        embeddings: numpy array of shape (num_samples, 512)
        labels: numpy array of shape (num_samples,)
        images: list of PIL images or raw tensors (for visualization)
    """
    model.eval()
    extractor = FeatureExtractor(model, layer_name=layer_name)
    
    all_labels = []
    all_images = []
    
    for x, y in loader:
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.squeeze(1)
            
        x_dev = x.to(device)
        model(x_dev)  # Forward pass triggers hook
        
        all_labels.extend(y.numpy())
        all_images.append(x)
        
    # Concatenate features
    features = torch.cat(extractor.features, dim=0) # Shape: (N, 512, 1, 1) or similar
    features = torch.flatten(features, 1).numpy()
    
    all_labels = np.array(all_labels)
    all_images = torch.cat(all_images, dim=0).numpy()
    
    extractor.close()
    return features, all_labels, all_images


def run_pca_reduction(embeddings, n_components=2):
    """Runs PCA on embeddings."""
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    return embeddings_reduced, pca.explained_variance_ratio_


def run_tsne_reduction(embeddings, n_components=2):
    """Runs t-SNE on embeddings."""
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_reduced = tsne.fit_transform(embeddings)
    return embeddings_reduced


def find_nearest_neighbors(query_idx: int, embeddings: np.ndarray, top_k: int = 5):
    """
    Finds the top-K nearest neighbors to a query image in the embedding space.
    Uses Euclidean distance.
    Returns:
        indices: list of nearest indices (excluding query_idx itself)
        distances: list of distances
    """
    query_emb = embeddings[query_idx]
    
    # Calculate Euclidean distance to all embeddings
    distances = np.linalg.norm(embeddings - query_emb, axis=1)
    
    # Sort indices by distance (excluding query itself which has dist = 0)
    sorted_indices = np.argsort(distances)
    
    neighbor_indices = []
    neighbor_distances = []
    
    for idx in sorted_indices:
        if idx == query_idx:
            continue
        neighbor_indices.append(idx)
        neighbor_distances.append(distances[idx])
        if len(neighbor_indices) == top_k:
            break
            
    return neighbor_indices, neighbor_distances

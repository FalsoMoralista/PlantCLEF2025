import faiss
import faiss.contrib.torch_utils

import torch
import numpy as np
np.random.seed(0)


class KMeansModule:

    def __init__(self, dimensionality=256, n_iter=500, tol=1e-4, K=2, resources=None, config=None):
        
        self.resources = resources
        self.config = config

        self.K = K
        self.d = dimensionality
        self.max_iter = n_iter
        self.tol = tol
        
        # Note: We're not using gpu parameter here since it's not compatible in some versions
        self.kmeans = faiss.Kmeans(
            d=self.d,           # dimension of the vectors
            k=self.K,           # number of clusters
            nredo=50,           # number of cluster redos
            niter=self.max_iter,  # number of iterations
            verbose=False        # verbosity level
        )
        

    def train(self, cached_features):
        """
        Train K-means on features from the feature bank using GPU if available
        
        Args:
            cached_features: Dictionary with class IDs as keys and lists of feature tensors as values
        
        Returns:
            Inertia (objective function) value
        """
        # Combine all features into a single tensor
        x = []
        for id in cached_features.keys():
            image_list = cached_features[id]
            x.append(torch.stack(image_list))
        x = torch.cat(x)#.to(torch.device('cuda:0'))

        print(f"Training K-means with {x.size()}")        
            
        # Set CPU index as the base for the GPU index
        cpu_index = faiss.IndexFlatL2(self.d)
        
        # Transfer to GPU
        gpu_index = faiss.index_cpu_to_gpu(self.resources, 0, cpu_index)
        
        # Use the GPU index in kmeans
        self.kmeans.index = gpu_index
        
        # Enable GPU clustering
        self.kmeans.gpu = True
                
        # Train K-means model
        self.kmeans.train(x)
        
        # Store the centroids
        self.centroids = torch.from_numpy(self.kmeans.centroids).float()
        
        # Return the inertia (final objective value)
        self.inertia_ = self.kmeans.obj[-1]
        return self.inertia_        

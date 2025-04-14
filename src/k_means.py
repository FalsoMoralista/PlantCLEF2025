import faiss
import faiss.contrib.torch_utils

import torch
import numpy as np
np.random.seed(0)


class KMeansModule:

    def __init__(self, dimensionality=256, n_iter=1000, tol=1e-4, K=2, resources=None, config=None):
        
        self.resources = resources
        self.config = config

        self.K = K
        self.d = dimensionality
        self.max_iter = n_iter
        self.tol = tol
        
        # Create the K-means object
        self.kmeans = faiss.Kmeans(d=self.d, k=self.K, nredo=10, niter=self.max_iter, verbose=True) # TODO: set verbose to false

    def train(self, cached_features):
        x = []
        for id in cached_features.keys():
            image_list = cached_features[id]
            x.append(torch.cat(image_list))
        x = torch.cat(x)
        print('x size', x.size())
        # Then train K-means model for one iteration to initialize centroids (kmeans++ init)
        self.kmeans.train(x.numpy()) 
        
        # Replace the regular index by a gpu one     
        #index_flat = self.n_kmeans[class_id][k].index
        #gpu_index_flat = faiss.index_cpu_to_gpu(self.resources, rank, index_flat)
        #self.n_kmeans[class_id][k].index = gpu_index_flat
        self.inertia_ = self.kmeans.obj[-1]
        return self.inertia_
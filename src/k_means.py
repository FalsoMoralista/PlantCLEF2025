import faiss
import faiss.contrib.torch_utils

import torch
import numpy as np
import itertools
import torch.nn.functional as F
np.random.seed(0)


class KMeansModule:

    def __init__(self, dimensionality=256, n_iter=350, tol=1e-4, K=2, resources=None, config=None):
        
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
            nredo=30,           # number of cluster redos
            niter=self.max_iter,  # number of iterations
            verbose=False,        # verbosity level
            min_points_per_centroid = self.K # avoiding warnings concerning minimum amount of points
        )
        

    def train_from_batch(self, x):
        """
            Train K-means on features from the feature bank using GPU, and store the L2 normalized
            prototypes for computing the Cosine Cluster Index metric.
        """
            
        # Set CPU index as the base for the GPU index
        cpu_index = faiss.IndexFlatL2(self.d)
        
        # Transfer to GPU
        gpu_index = faiss.index_cpu_to_gpu(self.resources, 0, cpu_index)
        
        # Use the GPU index in kmeans
        self.kmeans.index = gpu_index
        
        # Enable GPU clustering
        self.kmeans.gpu = True
                
        self.kmeans.train(x)
        
        # Normalize centroids
        normalized_centroids = F.normalize(torch.from_numpy(self.kmeans.centroids), p=2, dim=1)
        normalized_centroids_np = normalized_centroids.cpu().numpy().astype('float32')

        # Replace the index with a new one containing normalized centroids
        index = faiss.IndexFlatIP(self.d)  # Use Inner Product for cosine similarity
        gpu_index = faiss.index_cpu_to_gpu(self.resources, 0, index)

        # Add normalized centroids to the index
        gpu_index.add(normalized_centroids_np)

        # Replace the internal index
        self.kmeans.index = gpu_index
        self.kmeans.gpu = True

        # Store the centroids
        self.centroids = normalized_centroids.to(torch.device('cuda:0'))
        
        # Return the inertia (final objective value)
        self.inertia_ = self.kmeans.obj[-1]
        return self.inertia_        

    def train(self, cached_features):
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
        self.kmeans.gpu = True
        self.kmeans.train(x)
        
        # Store centroids as tensor
        self.centroids = torch.from_numpy(self.kmeans.centroids).float()
        
        self.inertia_ = self.kmeans.obj[-1]
        return self.inertia_        

def inter_cluster_separation(n_kmeans, k_range, device):
    
    S_scores = torch.zeros(len(k_range), device=device)
    
    for k_i, k in enumerate(k_range):
        centroids = n_kmeans[k_i].centroids

        pairs = torch.combinations(torch.arange(k, device=device), r=2)

        # Extract the centroid pairs in a single operation
        centroid_pairs = centroids[pairs]

        # Compute cosine similarity in a vectorized way
        cosine_similarities = F.cosine_similarity(centroid_pairs[:, 0], centroid_pairs[:, 1], dim=1) # F.pairwise_distance(centroid_pairs[:, 0], centroid_pairs[:, 1])
        
        # Accumulate the sum of cosine similarities
        S_scores[k_i] = cosine_similarities.sum() / len(pairs) # averaging by the number of combinations because in its original formulation it favours large values of K
    return S_scores

def cosine_cluster_index(n_kmeans, k_range, xb, device):
    batch_size = xb.size(0)
    
    xb = F.normalize(xb, p=2, dim=1)  # shape: [batch_size, dim]

    best_K_values = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    cluster_assignments = torch.zeros(batch_size, dtype=torch.int32, device=device)

    S_scores = inter_cluster_separation(n_kmeans=n_kmeans,
                                                k_range=k_range,
                                                device=device)

    C_scores = torch.zeros(len(k_range), device=device)
    for k_i, k_range in enumerate(k_range):
        D, batch_assignments = n_kmeans[k_i].kmeans.index.search(xb.contiguous(), 1) # for each k compute the cluster assignments for xb (x_batch)
        batch_assignments = batch_assignments.squeeze(-1)

        centroids = n_kmeans[k_i].centroids

        centroid_list = centroids[batch_assignments] # Retrieve centroids from batch assignments

        # Compute the cosine similarity between every image and the cluster centroid to which is associated to
        C_score = F.cosine_similarity(xb, centroid_list) # F.pairwise_distance(batch_x, centroid_list) 
        C_scores[k_i] = C_score.sum()

    CCI = S_scores / (C_scores + S_scores)
    best_K_values = CCI.argmax().item()

    # Find the cluster assignment for the best K value
    D, c_assignment = n_kmeans[best_K_values].kmeans.index.search(xb.contiguous(), 1)
    c_assignment = c_assignment.squeeze(-1)

    cluster_assignments = c_assignment

    return best_K_values, cluster_assignments
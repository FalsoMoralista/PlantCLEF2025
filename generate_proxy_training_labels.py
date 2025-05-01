from urllib.request import urlopen
import torch
import timm
import faiss
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from src.helper import (load_class_mapping, load_species_mapping)
from src.k_means import KMeansModule

from datasets.pc2025 import build_train_dataset 
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args):

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        print('Device found')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    
    cache_dir = '/home/rtcalumby/adam/luciano/PlantCLEF2025/cached_features/'
    cache_lists = [torch.load(cache_dir + f'feature_bank_rank{r}.pt') for r in range(2)]
    # Merge dictionaries
    feature_bank = {}
    for cache in cache_lists:
        for key in cache.keys():
            if key not in feature_bank:
                feature_bank[key] = []
            # Extend the list with all tensors for this key from the current cache
            feature_bank[key].extend(cache[key])
    cnt = [len(feature_bank[key]) for key in feature_bank.keys()]    
    print('Total features gathered %d, problem size: %d' % (sum(cnt), 1408034))
    assert sum(cnt) == 1408034, 'Cache not compatible, corrupted or missing'

    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = 0

    k_means = KMeansModule(K=7806, dimensionality=768, config=config, resources=resources)
    energy = k_means.train(cached_features=feature_bank)
    print('K-Means free energy', energy)
    print('Kmeans centroids', k_means.centroids.size())
    torch.save('proxy/prototypes.pt')  

main(None)
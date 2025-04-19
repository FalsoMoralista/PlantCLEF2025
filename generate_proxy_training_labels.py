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
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    
    # -- Model
    pretrained_path = args['pretrained_path']
    # -- Data
    base_dir = args['base_dir']
    train_dir = args['data']['train_data']
    cache_dir = args['data']['output_dir']
    class_mapping = args['data']['class_mapping']
    species_mapping = args['data']['species_mapping']
    # --

    cid_to_spid = load_class_mapping(class_mapping)
    spid_to_sp = load_species_mapping(species_mapping)

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
    model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
    model = model.to(device)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    batch_size = 2048
    cache_lists = [torch.load(cache_dir + f'feature_bank_rank{r}.pt') for r in range(8)]
    feature_bank = {}
    for cache in cache_lists:
        for key, tensor_list in cache:
            feature_bank.setdefault(key, []).append(tensor_list)
    feature_bank = {key: torch.cat(tensors, dim=0) for key, tensors in feature_bank.items()}
    
    cnt = [len(feature_bank[key]) for key in feature_bank.keys()]    
    print('Total features gathered %d, problem size: %d' % (sum(cnt), 1408033))
    assert sum(cnt) == 1408033, 'Cache not compatible, corrupted or missing'

    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = 0

    k_means = KMeansModule(K=7806, dimensionality=768, config=config, resources=resources)
    energy = k_means.train(cached_features=feature_bank)
    print('K-Means free energy', energy)

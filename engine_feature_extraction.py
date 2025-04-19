import os
# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
except Exception:
    pass

from urllib.request import urlopen
import torch
import timm
import faiss
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from src.helper import (load_class_mapping, load_species_mapping)
from datasets.pc2025 import build_train_dataset 
import logging
import sys
import multiprocessing as mp
from util.distributed import init_distributed
import torch.distributed as dist

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

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)    

    cid_to_spid = load_class_mapping(class_mapping)
    spid_to_sp = load_species_mapping(species_mapping)

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
    model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
    model = model.to(device)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    batch_size = 2048
    train_dataset, train_dataloader, dist_sampler = build_train_dataset(image_folder=train_dir,
                                                data_config=data_config,
                                                num_workers=4,
                                                world_size=world_size,
                                                rank=rank,
                                                batch_size=batch_size)
    items = []
    for im_id, (batch_data, labels, img_name) in enumerate(train_dataloader):
        logger.info(f'Iteration [{im_id+1}/{len(train_dataset)/(batch_size * world_size)}]')
        x = batch_data.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            with torch.inference_mode():
                output = model(x).to(device=torch.device('cpu'))
        if (im_id + 1) % 2 == 0:
            torch.cuda.empty_cache()
        items.append((output, labels))
    # -- End 
    def build_cache():
        feature_bank = {}        
        for outputs, targets in items:
            for x, y in zip(outputs, targets):
                class_id = y.item()
                if not class_id in feature_bank:
                    feature_bank[class_id] = []                    
                feature_bank[class_id].append(x)
        return feature_bank    
    feature_bank = build_cache()
    torch.save(feature_bank, cache_dir + f'feature_bank_rank{rank}.pt')  
    dist.barrier()
    print('Rank,', rank, 'Cache keys', len(feature_bank.keys()))
    # -- 
    # Assert everything went fine
    if rank == 0:
        dataset_length = 1408033
        cache_list = [torch.load(cache_dir + f'feature_bank_rank{r}.pt') for r in range(world_size)]
        total = 0
        for cache in cache_list:
            cnt = [len(cache[key]) for key in cache.keys()]
            total += sum(cnt)
        logger.info('Total features gathered %d, problem size: %d' % (total, dataset_length))
        assert total == dataset_length, 'Cache not compatible, corrupted or missing'
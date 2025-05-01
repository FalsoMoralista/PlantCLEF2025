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
from src.k_means import KMeansModule
from datasets.pc2025 import build_test_dataset, build_test_transform
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
        device = torch.device('cuda:0') # this is overwritten by the init_distributed() function (which assigns the corresponding rank for each process)
        torch.cuda.set_device(device)
    
    # -- Model
    pretrained_path = args['pretrained_path']
    
    # -- Clustering 
    K = args['k_means']['K']

    # -- Patch configuration
    N = args['patches']['N']

    # -- Data
    base_dir = args['base_dir']
    test_dir = args['data']['test_data']
    class_mapping = args['data']['class_mapping']
    species_mapping = args['data']['species_mapping']
    # --

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend (overwrites manual replacement of ranks)
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

    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = rank

    for n in N:
        if rank == 0:
            k_means = KMeansModule(K=K, dimensionality=768, config=config, resources=resources)

        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        
        test_dataset, test_dataloader = build_test_dataset(image_folder=test_dir,
                                                    data_config=data_config,
                                                    num_workers=4,
                                                    n=n,
                                                    batch_size=1)
        batch_size = 1024
        feature_bank = {}
        for im_id, (preprocessed_images, _, name) in enumerate(test_dataloader):
            logger.info(f'Image [{im_id+1}/2105]')
            
            id = name[0].replace('.jpg','')   
            feature_bank[id] = []

            patches = preprocessed_images.squeeze(0) # 8192                         
            patch_dataset = torch.utils.data.TensorDataset(patches)
            patch_sampler = DistributedSampler(
                patch_dataset, 
                num_replicas=world_size,
                rank=rank,
                shuffle=False  
            )            
            patch_loader = DataLoader(
                patch_dataset,
                batch_size=batch_size,
                sampler=patch_sampler,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )                
            
            for i, batch in enumerate(patch_loader):
                x = batch[0].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    with torch.inference_mode():
                        output = model(x).to(device=torch.device('cpu'))
                feature_bank[id].append(output)

        # -- Gathering 
        if world_size > 1:
            # Convert cache to list format for gathering
            local_cache_list = [(key, torch.stack(value)) for key, value in feature_bank.items()]
            # Gather cache lists from all processes
            all_cache_lists = [None for _ in range(world_size)]
            dist.all_gather_object(all_cache_lists, local_cache_list) 

            aggregated_cache = {}
            if rank == 0:
                for cache_list in all_cache_lists:
                    for key, tensor_list in cache_list:
                        aggregated_cache.setdefault(key, []).append(tensor_list)
                aggregated_cache = {key: torch.cat(tensors, dim=0) for key, tensors in aggregated_cache.items()}                
            dist.barrier()
            feature_bank = aggregated_cache
        # -- 
        # Assert everything went fine
        if rank == 0:
            problem_size = (2048/n)*(1024/n) * len(test_dataset)
            cnt = [len(feature_bank[key]) for key in feature_bank.keys()]    
            logger.info('Total features gathered %d, problem size: %d' % (cnt, problem_size))
            assert sum(cnt) == problem_size, 'Cache not compatible, corrupted or missing'
            energy = k_means.train(cached_features=feature_bank)
            print('K-Means free energy', energy, 'n:', n)

    #img = None
    #if 'https://' in args.image or 'http://' in  args.image:
    #    img = Image.open(urlopen(args.image))
    #elif args.image != None:
    #    img = Image.open(args.image)
        
    #if img != None:
    #    img = transforms(img).unsqueeze(0)
    #    img = img.to(device)
    #    output = model(img)  # unsqueeze single image into batch of 1
    #    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    #    top5_probabilities = top5_probabilities.cpu().detach().numpy()
    #    top5_class_indices = top5_class_indices.cpu().detach().numpy()

    #    for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
    #        species_id = cid_to_spid[cid]
    #        species = spid_to_sp[species_id]
    #        print(species_id, species, proba)

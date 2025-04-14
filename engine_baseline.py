from urllib.request import urlopen
import os
import torch
import timm
import faiss
from torch.utils.data import DataLoader
from PIL import Image
from src.helper import (load_class_mapping, load_species_mapping)
from src.k_means import KMeansModule
from datasets.pc2025 import build_test_dataset, build_test_transform

def load_images():
    return 0

def main(args):
    
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


    cid_to_spid = load_class_mapping(class_mapping)
    spid_to_sp = load_species_mapping(species_mapping)
        
    rank = 0
    device = torch.device(args['devices'][rank])    

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
    model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
    model = model.to(device)
    model = model.eval()

    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = rank
    for n in N:
        k_means = KMeansModule(K=K, dimensionality=768, config=config, resources=resources)

        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        
        test_dataset, test_dataloader = build_test_dataset(image_folder=test_dir,
                                                    data_config=data_config,
                                                    num_workers=16,
                                                    n=n,
                                                    batch_size=1)
        batch_size = 1024
        feature_bank = {}
        for im_id, (preprocessed_images, _, name) in enumerate(test_dataloader):
            print(f'Image [{im_id+1}/2105]', flush=True)
            id = name[0].replace('.jpg','')   
            feature_bank[id] = []
            loader = DataLoader(preprocessed_images.squeeze(0), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32, pin_memory=True)
            for i, batch in enumerate(loader):
                x = batch.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    with torch.inference_mode():
                        output = model(x).to(device=torch.device('cpu'))
                feature_bank[id].append(output)                    
        # -- 
        # Assert everything went fine
        cnt = [len(feature_bank[key]) for key in feature_bank.keys()]    
        problem_size = (2048/n)*(1024/n) * len(test_dataset)
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

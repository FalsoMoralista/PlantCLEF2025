from urllib.request import urlopen
from PIL import Image
import timm
import torch
from src.helper import (load_class_mapping, load_species_mapping)
from datasets import build_test_dataset

def main(args):
    
    # -- Model
    base_dir = args['base_dir']
    pretrained_path = args['pretrained_path']
    
    # -- Classification
    class_mapping = args['data']['class_mapping']
    species_mapping = args['data']['species_mapping']

    # -- Clustering 
    K = args['k_means']['K']

    # -- Patch configuration
    N = args['patches']['N']

    cid_to_spid = load_class_mapping(class_mapping)
    spid_to_sp = load_species_mapping(species_mapping)
        
    device = torch.device(args['devices'][0])    
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
    model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
    
    model = model.to(device)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    print('Data config:', data_config)
    test_dataset, loader_test, sampler_test = build_test_dataset(is_train=False, data_config=data_config)
    
    #transforms = timm.data.create_transform(**data_config, is_training=False)



    img = None
    if 'https://' in args.image or 'http://' in  args.image:
        img = Image.open(urlopen(args.image))
    elif args.image != None:
        img = Image.open(args.image)
        
    if img != None:
        img = transforms(img).unsqueeze(0)
        img = img.to(device)
        output = model(img)  # unsqueeze single image into batch of 1
        top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
        top5_probabilities = top5_probabilities.cpu().detach().numpy()
        top5_class_indices = top5_class_indices.cpu().detach().numpy()

        for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
            species_id = cid_to_spid[cid]
            species = spid_to_sp[species_id]
            print(species_id, species, proba)

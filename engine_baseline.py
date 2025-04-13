from urllib.request import urlopen
from PIL import Image
import timm
import torch
from src.helper import (load_class_mapping, load_species_mapping, build_test_transform)
from datasets import build_test_dataset
import os

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
        
    device = torch.device(args['devices'][0])    
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
    model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
    
    model = model.to(device)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    
    crop_and_resize = build_test_transform(data_config, n=N[0])    
    image_list = os.listdir(base_dir+test_dir)
    
    feature_bank = {}
    for image in image_list:
        print('Img name:', image)
        im = Image.open(test_dir+image)
        im = crop_and_resize(im)
        break        




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

import torch
import csv
import time
import sys
sys.path.append('../')

from collections import Counter
from src.helper import load_class_mapping, load_species_mapping
cid_to_spid = load_class_mapping('/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/class_mapping.txt')
spid_to_sp = load_species_mapping('/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/species_id_to_name.txt')

thresholds = [0.6]
for threshold in thresholds:
    logits = torch.load(f'logits/clustered_preds_run_0_ep15_treshold={threshold}.pt')
    predicted_items = []
    for key in logits.keys():
        K = len(logits[key]) # Best K for each test image
        image_predictions = []
        print('K', K)
        for k in range(K): # For each cluster of patches
            votes = []
            out = logits[key][k]  
            for output in out:
                top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=0) * 100, k=5)
                top5_probabilities = top5_probabilities.to(torch.float32).cpu().detach().numpy()
                top5_class_indices = top5_class_indices.to(torch.float32).cpu().detach().numpy()
                proba, cid = top5_probabilities[0], top5_class_indices[0]
                species_id = cid_to_spid[cid]
                species = spid_to_sp[species_id]
                print('prob:', proba)
                votes.append(species_id)
            counter = Counter(votes) # FIXME use argmax 
            most_common_element, count = counter.most_common(1)[0]
            species_id = most_common_element
            print('speci', species_id)
            print('votes:',votes)
            
            if not species_id in image_predictions:
                image_predictions.append(species_id)      
        print('image pred', image_predictions)      
        exit(0)    
        predicted_items.append((key , image_predictions))
    print('Predicted items length:', len(predicted_items))    
    with open(f'submissions/clustered_pred_run_0_ep15_treshold={threshold}.csv', mode='w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['quadrat_id', 'species_ids'])  # Header
        for quadrat_id, species_ids in predicted_items:
            writer.writerow([quadrat_id, str(species_ids)])
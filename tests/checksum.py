import torch
import os

world_size = [0,1]
dataset_length = 1408034
cache_dir = '/home/rtcalumby/adam/luciano/PlantCLEF2025/cached_features/'
cache_list = [torch.load(cache_dir + f'feature_bank_rank{r}.pt') for r in world_size]
total = 0
for cache in cache_list:
    cnt = [len(cache[key]) for key in cache.keys()]
    total += sum(cnt)
print('Total features gathered %d, problem size: %d' % (total, dataset_length))
assert total == dataset_length, 'Cache not compatible, corrupted or missing'
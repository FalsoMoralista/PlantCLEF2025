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
from src.helper import (load_class_mapping, load_species_mapping, init_opt)
from src.k_means import KMeansModule
from datasets.pc2025 import build_test_dataset, build_test_transform
import logging
from src.utils.logging import (gpu_timer,
    grad_logger,
    CSVLogger,
    AverageMeter)

import sys
import multiprocessing as mp
import torch.distributed as dist
from util.distributed import init_distributed, AllReduce
from src.models.custom_vision_transformer import PilotVisionTransformer
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import numpy as np

# --
log_timings = True
log_freq = 10
checkpoint_freq = 5
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

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
    
    # -- Optimizations
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    experiment_code = args['experiment_code']
    
    # -- 
    accum_iter = args['gradient_accumulation'] # Gradient accumulation
    batch_size = args['batch_size']

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend (overwrites manual replacement of ranks)
    world_size, rank = init_distributed() 
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        print('Rank', rank)
        logger.setLevel(logging.ERROR)    

    # -- make csv_logger
    experiment_dir = f'/home/rtcalumby/adam/luciano/PlantCLEF2025/logs/experiment_{experiment_code}/'
    log_file = experiment_dir + f'experiment_{experiment_code}_r{rank}.csv'
    latest_path = experiment_dir +f'experiment_{experiment_code}_latest.pth.tar'
    
    save_path = os.path.join(experiment_dir, f'experiment_{experiment_code}' + '-ep{epoch}.pth.tar')


    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'Train loss'),
                           ('%d', 'time (ms)'))

    cid_to_spid = load_class_mapping(class_mapping)
    spid_to_sp = load_species_mapping(species_mapping)


    test_dir = '/home/rtcalumby/adam/luciano/PlantCLEF2025/test_dataset/'

    patch_size = N
    log_freq = accum_iter

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
    model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
    model.to(device)
    data_config = timm.data.resolve_model_data_config(model)

    test_dataset, test_dataloader, dist_sampler = build_test_dataset(image_folder=test_dir,
                                                data_config=data_config,
                                                input_resolution=(3072,2048),
                                                num_workers=12,
                                                n=patch_size,
                                                world_size=world_size,
                                                rank=rank,
                                                shuffle=True,
                                                batch_size=batch_size)
    use_bfloat16 = True
    ipe = len(test_dataloader)
    print('Test dataset, length:', ipe * batch_size)
    ViT = PilotVisionTransformer(img_size=[3072,2048], pretrained_patch_embedder=model, patch_size=patch_size, embed_dim=768, depth=6)
    logger.info('Loading Vision Transformer: %s' %(ViT))
    ViT.to(device)
    
    # Create optimizer and config model
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=ViT,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)

    ViT = DistributedDataParallel(ViT, static_graph=True, find_unused_parameters=False)
    ViT_noddp = ViT.module
    print('Allocated Memory with model loading:', (torch.cuda.memory_allocated() / 1024.**3), ' GB')

    prototypes = torch.load('proxy/prototypes.pt').to(device)
    
    def save_checkpoint(epoch):
        save_dict = {
            'custom_vit': ViT_noddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr # TODO if loading from preemption readjust lr and optmizer settings to correspondent stopping epoch
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    
    # -- TRAINING LOOP
    start_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        
        dist_sampler.set_epoch(epoch) 

        logger.info('Epoch %d' % (epoch + 1))
        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (preprocessed_images, _, name) in enumerate(test_dataloader):
            
            def train_step():
                _new_lr = optimizer.param_groups[0]['lr']
                _new_wd = optimizer.param_groups[0]['weight_decay']
                grad_stats = None

                if (itr + 1) % accum_iter == 0:                
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()

                x = preprocessed_images.to(device, non_blocking=True)
                
                y = prototypes

                def criterion(z, h):
                    loss = F.mse_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    output = ViT(x.squeeze(0)).squeeze(0)
                    loss = criterion(output.T, y)

                loss_meter.update(loss)
                
                #  Step 2. Backward & step with mixed precision
                if use_bfloat16:
                    scaler(loss, optimizer, clip_grad=None,
                                parameters=(ViT_noddp.parameters()),
                                create_graph=False, retain_graph=False,
                                update_grad=(itr + 1) % accum_iter == 0)
                else:
                    loss.backward()
                    optimizer.step()
                
                if (itr + 1) % accum_iter == 0:
                    grad_stats = grad_logger(ViT_noddp.named_parameters())
                    optimizer.zero_grad()
                    
                return (float(loss), _new_lr , _new_wd, grad_stats)

            # -- 
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            
            time_meter.update(etime)

             # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d/%5d] - train_loss: %.4f -'
                                '[wd: %.2e] [lr: %.2e]'
                                '[max mem allocated: %.2e] '
                                '(%.1f ms)'

                                % (epoch + 1, itr, ipe,
                                    loss_meter.avg,
                                    _new_wd,
                                    _new_lr,
                                    torch.cuda.max_memory_allocated() / 1024.**2,
                                    time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                        grad_stats.first_layer,
                                        grad_stats.last_layer,
                                        grad_stats.min,
                                        grad_stats.max))
            log_stats()                       
        # -- end of epoch
        save_checkpoint(epoch+1)
        logger.info('Loss %.4f' % loss)        


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

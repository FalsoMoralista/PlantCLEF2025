import argparse
import yaml
import pprint
from engine_baseline import main as app_main
import multiprocessing as mp
from util.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='config/configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')    

parser.add_argument("--image", type=str, default='https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata/test/1361687/bd2d3830ac3270218ba82fd24e2290becd01317c.jpg') #Orchis simia
#bd2d3830ac3270218ba82fd24e2290becd01317c.jpg

def process_main(rank, fname, world_size, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')    

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        logger.info(params)
    params['devices'] = devices
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')    
    app_main(args=params)
    
if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()

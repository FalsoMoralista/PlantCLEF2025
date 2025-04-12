import argparse
import yaml
import pprint
from engine_baseline import main as app_main

if __name__ == '__main__':

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

    args = parser.parse_args()
    # -- load script params
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    params['devices'] = args.devices
    app_main(args=params)
    
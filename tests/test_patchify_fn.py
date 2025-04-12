from PIL import Image
import os

import sys
# setting path to the repository root
sys.path.append('../')

def main():
    test_dir = '/home/rtcalumby/adam/luciano/PlantCLEF2025/test_dataset/'
    image_list = os.listdir(test_dir)
    print('Images:', len(image_list))
    for image in image_list:
        print(Image.open(test_dir+image))

main()
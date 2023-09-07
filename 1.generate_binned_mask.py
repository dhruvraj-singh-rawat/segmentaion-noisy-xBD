import os
import cv2
import numpy as np 
import random
import numpy as np
from PIL import Image

dataset_type = 'train' #test,eval

output_target_dir = dataset_type+'/binned_targets'
image_target_dir = dataset_type+'/targets'

def bin_images(image):
    # Convert pixel values less than 2 to 0 and values greater than or equal to 2 to 1
    image[image < 2] = 0
    image[image > 2] = 255
    
    return image

def generate_noisy_images(image_target_dir,output_target_dir):


    output_dir_mask = os.path.join('', output_target_dir)
    os.makedirs(output_dir_mask, exist_ok=True)


    # List all image files in the directory
    image_files = [f for f in os.listdir(image_target_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    for image_file in image_files:
        
        image_mask_path = os.path.join(image_target_dir, image_file)
        mask = cv2.imread(image_mask_path)

        binned_mask = bin_images(mask)

        output_path_mask = os.path.join(output_target_dir, image_file)
        cv2.imwrite(output_path_mask, binned_mask)

generate_noisy_images(image_target_dir,output_target_dir)
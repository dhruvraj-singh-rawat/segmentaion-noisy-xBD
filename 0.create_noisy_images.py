import os
import cv2
import numpy as np 
import random
import numpy as np
from PIL import Image
from scipy.ndimage.filters import generic_filter

# Input Image Dir
image_dir = "train/images/"
image_target_dir = "train/targets/"

# Output Dir
output_image_dir = "blurred_images/train/images/"
output_target_dir = "blurred_images/train/targets/"

##############################################################################
# Noise Parameters 

gaussian_stddev = 0.35
salt_pepper_probability = 0.002
haze_factor = 0.15

from PIL import Image, ImageFilter

def majority_filter(mask, neighborhood_size, threshold):
    def majority_value(arr):
        unique_classes, counts = np.unique(arr, return_counts=True, axis=0)
        max_count = np.max(counts)
        if max_count >= threshold:  # Check if the max count crosses the threshold
            majority_class = unique_classes[np.argmax(counts)]
            return majority_class
        else:
            return arr[int(len(arr)/2)]  # Keep the original value

    filtered_mask = np.zeros_like(mask)

    for channel in range(mask.shape[2]):
        channel_filtered = generic_filter(mask[:, :, channel], majority_value, size=neighborhood_size)
        filtered_mask[:, :, channel] = channel_filtered

    return filtered_mask

def bin_images(image):
    # Convert pixel values less than 2 to 0 and values greater than or equal to 2 to 1
    image[image < 2] = 0
    image[image >= 2] = 1
    
    return image
    
    
def generate_noisy_images(input_dir,image_target_dir,
                          output_image_dir,output_target_dir,
                          gaussian_stddev,salt_pepper_probability,
                          haze_factor,
                         perform_majority_filtering,
                         bin_mask = False,
                         early_stopping = False):

    # Create separate folders for different types of noise
    output_dir_image = os.path.join('', output_image_dir)
    output_dir_mask = os.path.join('', output_target_dir)


    os.makedirs(output_dir_image, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)


    # List all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image1 = Image.open(image_path)
        
        # Read Mask
        image_mask_path = os.path.join(image_target_dir, image_file[:-4] + '_target.png')
        mask = cv2.imread(image_mask_path)
        #mask = Image.open(image_mask_path)
        if bin_mask:
            mask = bin_images(mask)
        
        if perform_majority_filtering:
            mask = majority_filter(mask, neighborhood_size, threshold)
        
        print(mask.shape)

        # Add Gaussian noise
        mean = 0
        stddev = gaussian_stddev
        gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        noisy_image_gaussian = cv2.add(image, gaussian_noise)

        # Add salt and pepper noise
        salt_pepper_noise = np.zeros(image.shape, np.uint8)
        probability = salt_pepper_probability
        salt = np.where(np.random.rand(*image.shape[:2]) < probability)
        pepper = np.where(np.random.rand(*image.shape[:2]) < probability)
        salt_pepper_noise[salt] = 255
        salt_pepper_noise[pepper] = 0
        noisy_image_salt_pepper = cv2.add(image, salt_pepper_noise)

        # Generate Hazy image 
        hazed_image = image1.filter(ImageFilter.GaussianBlur(radius=haze_factor * 10))


        # Save the noisy images in separate folders
        output_path_gaussian = os.path.join(output_dir_image, image_file[:-4]+'_gaussian'+image_file[-4:])
        output_path_salt_pepper = os.path.join(output_dir_image, image_file[:-4]+'_salt_pepper'+image_file[-4:])
        output_path_gaussian_blur = os.path.join(output_dir_image, image_file[:-4]+'_gaussian_blur'+image_file[-4:])
        output_path_original = os.path.join(output_dir_image, image_file)
        
        # Mask for different Noise
        output_path_gaussian_mask = os.path.join(output_dir_mask, image_file[:-4]+'_gaussian_target'+image_file[-4:])
        output_path_salt_pepper_mask = os.path.join(output_dir_mask, image_file[:-4]+'_salt_pepper_target'+image_file[-4:])
        output_path_gaussian_blur_mask = os.path.join(output_dir_mask, image_file[:-4]+'_gaussian_blur_target'+image_file[-4:])
        output_path_original_mask = os.path.join(output_dir_mask, image_file[:-4]+'_target'+image_file[-4:])
        
        # Saving
        print(output_path_gaussian)
        cv2.imwrite(output_path_gaussian, noisy_image_gaussian)
        cv2.imwrite(output_path_salt_pepper, noisy_image_salt_pepper)
        hazed_image.save(output_path_gaussian_blur)
        cv2.imwrite(output_path_original, image)
        
        print(output_path_gaussian_mask)
        
        cv2.imwrite(output_path_gaussian_mask,mask)
        cv2.imwrite(output_path_salt_pepper_mask,mask)
        cv2.imwrite(output_path_gaussian_blur_mask,mask)
        cv2.imwrite(output_path_original_mask,mask)
        
#         mask.save(output_path_gaussian_mask)
#         mask.save(output_path_salt_pepper_mask)
#         mask.save(output_path_original_mask)
        
        if early_stopping:
            break

    print("Noisy images saved.")


def load_data(data_dir, resize=False):
    # Set the paths to the directories containing the dataset
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'targets')

    
    # Set the input image dimensions
    input_shape = (128, 128, 3)  # Adjust as needed

    # Get the list of image filenames
    image_filenames = os.listdir(images_dir)

    # Batch size for loading images
    batch_size = 32

    # Prepare the data for training
    X_train = []
    y_train = []

    # Process images in batches
    for i in range(0, len(image_filenames), batch_size):
        batch_filenames = image_filenames[i:i + batch_size]
        batch_X = []
        batch_y = []

        for filename in batch_filenames:
            # Load pre and post-disaster images
            pre_image_path = os.path.join(images_dir, filename)
            pre_image = cv2.imread(pre_image_path)

            # Load the corresponding label
            label_filename = filename[:-4] + '_target.png'
            label_path = os.path.join(labels_dir, label_filename)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if resize:
                # Resize the images to the desired shape
                pre_image = cv2.resize(pre_image, (input_shape[0], input_shape[1]))
                label = cv2.resize(label, (input_shape[0], input_shape[1]))

            # Add the data to the batch
            batch_X.append(pre_image)
            batch_y.append(label)

        # Convert the batch lists to numpy arrays
        batch_X = np.array(batch_X)
        print(len(batch_y))

        # Check if all labels in the batch have the same shape
        label_shapes = set([label.shape for label in batch_y])
        if len(label_shapes) > 1:
            raise ValueError("Labels in the batch have different shapes.")

        batch_y = np.array(batch_y)

        # Add the batch data to the training set
        X_train.append(batch_X)
        y_train.append(batch_y)

    # Concatenate the batches to obtain the final training data
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    return X_train, y_train

##############################################################################

generate_noisy_images(image_dir,image_target_dir,
                      output_image_dir,output_target_dir,
                      gaussian_stddev,salt_pepper_probability,
                      haze_factor,
                     perform_majority_filtering = 0 ,
                     bin_mask = False,
                     early_stopping = False)



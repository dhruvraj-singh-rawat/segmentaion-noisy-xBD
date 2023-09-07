import os
import cv2
import numpy as np
import json

dataset_type = 'train' #test,eval

# Path to the folder containing binary masks
mask_folder = dataset_type+'/binned_targets'
# Save the COCO JSON file
output_json_path = dataset_type+'_output_coco_annotations.json'

# Initialize COCO annotations
coco_annotations = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "damage"}]
}

image_id = 1
annotation_id = 1

# Loop through the mask files in the folder
for mask_file in os.listdir(mask_folder):
    if mask_file.endswith('.png'):
        mask_path = os.path.join(mask_folder, mask_file)
        
        # Load the binary mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a polygon from the contour
        segmentations = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                segmentations.append(contour)
        
        # Add image information to annotations
        image_info = {
            "id": image_id,
            "file_name": mask_file[:-11]+'.png',  # Removed Targets from file_name
            "height": mask.shape[0],
            "width": mask.shape[1]
        }
        coco_annotations["images"].append(image_info)
        
        # Add annotation information to annotations
        for segmentation in segmentations:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Damage category
                "iscrowd": 0,
                "area": cv2.contourArea(np.array(segmentation).reshape(-1, 2)),
                "bbox": cv2.boundingRect(np.array(segmentation).reshape(-1, 2)),
                "segmentation": [segmentation],
            }
            coco_annotations["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1


with open(output_json_path, 'w') as json_file:
    json.dump(coco_annotations, json_file)

print(f"Saved COCO annotations to {output_json_path}")


import boto3
import csv
import cv2
import numpy as np
from omegaconf import OmegaConf
import os
import utils

import inference
from vit_model import ViTPoolClassifier

### Get and process the data
# Get the test antibodies
# This file is copied from the subcell-analysis repo
with open("test_antibodies.txt", "r") as f:
    test_antibodies = f.read().splitlines()

# Get the metadata associated with the test antibodies
# This file is copied from s3://czi-subcell-public/hpa-processed/cell_crops/metadata.csv
print("Loading metadata...")
metadata = {}
with open("metadata.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    # Convert header to a dictionary with column names as keys and indices as values for easier access
    header = {name: index for index, name in enumerate(header)}
    for row in reader:
        key = row[header["antibody"]]
        if key not in metadata:
            metadata[key] = [row]
        else:
            metadata[key].append(row)

### Get and process the cell images
s3 = boto3.client("s3")
s3_bucket = "czi-subcell-public"
cell_images = []
for antibody in test_antibodies:
    rows = metadata[antibody]
    for row in rows:
        if_plate_id = row[header["if_plate_id"]]
        position = row[header["position"]]
        sample = row[header["sample"]]
        cell_id = int(float(row[header["cell_id"]]))

        prefix = f"hpa-processed/cell_crops/{if_plate_id}/{if_plate_id}_{position}_{sample}_{cell_id}"
        cell_image_path = f"{prefix}_cell_image.png"
        cell_mask_path = f"{prefix}_cell_mask.png"

        # TODO: remove
        # Print the actual locations
        print(f"Cell location: {row[header['locations']]}")

        # If the cropped image already exists locally, load it and skip the download
        cropped_image_name = f"{if_plate_id}_{position}_{sample}_{cell_id}_cropped.png"
        if os.path.exists(cropped_image_name):
            print(f"Cropped image {cropped_image_name} already exists, skipping download.")
            cropped_image = cv2.imread(cropped_image_name, cv2.IMREAD_UNCHANGED)
            break

        print(f"Loading {cell_image_path} and {cell_mask_path} from S3...")
        # Read the cell image and cell mask from s3
        cell_image = utils.read_image_from_s3(s3, s3_bucket, cell_image_path)
        cell_mask = utils.read_image_from_s3(s3, s3_bucket, cell_mask_path)

        print(f"Processing {cell_image_path}...")
        # Apply the cell mask to the cell image
        masked_image = cv2.bitwise_and(cell_image, cell_image, mask=cell_mask)

        # Crop the image to 640x640 around the provided x1, x2, y1, y2 coordinates
        x1 = int(float(row[header["x1"]]))
        x2 = int(float(row[header["x2"]]))
        y1 = int(float(row[header["y1"]]))
        y2 = int(float(row[header["y2"]]))
        cropped_image = utils.crop_image(masked_image, x1, x2, y1, y2, crop_size=640)

        # Save the cropped image
        cropped_image_name = f"{if_plate_id}_{position}_{sample}_{cell_id}_cropped.png"
        cv2.imwrite(cropped_image_name, cropped_image)
        print(f"Saved processed image to {cropped_image_name}")
        break
    break

# TODO: run inference on a batch of images
### Run inference
# Get the model config
# TODO: generalize to allow configuration of channels and vit vs mae model type
print("Loading model configuration...")
# This file is copied from the SubCellPortable repo
with open("rybg_mae_model_config.yaml", "r") as f:
    model_config_file = OmegaConf.load(f)
    config = OmegaConf.to_container(model_config_file)

classifier_paths = None
if "classifier_paths" in config:
    classifier_paths = config["classifier_paths"]
encoder_path = config["encoder_path"]

# Download the model weights from s3
# If the model weights are already downloaded, skip this step
encoder_local_path = "rybg_mae_encoder.pth"
if os.path.exists(encoder_local_path):
    print("Encoder weights already downloaded, skipping download.")
else:
    print(f"Downloading encoder weights from S3: {encoder_path}")
    utils.download_s3_file(s3, s3_bucket, encoder_path, encoder_local_path)

# TODO: allow multiple classifiers
classifier_local_paths = "rybg_mae_classifier.pth"
if os.path.exists(classifier_local_paths):
    print("Classifier weights already downloaded, skipping download.")
else:
    print(f"Downloading classifier weights from S3: {classifier_paths[0]}")
    utils.download_s3_file(s3, s3_bucket, classifier_paths[0], classifier_local_paths)

# Load the model
print("Loading model...")
model_config = config.get("model_config")
model = ViTPoolClassifier(model_config)
model.load_model_dict(encoder_local_path, [classifier_local_paths])

# Run inference
print("Running inference...")
print(f"Input image shape: {cropped_image.shape}")
# Reshape image from (640, 640, 4) -> (4, 640, 640) for the model
cropped_image = np.transpose(cropped_image, (2, 0, 1))
print(f"Reshaped image shape: {cropped_image.shape}")
embedding, probabilities = inference.run_model(model, cropped_image, "output")
print(f"Embedding shape: {embedding.shape}")

# Get predicted classes
if classifier_paths:
    curr_probs_l = probabilities.tolist()
    max_location_class = curr_probs_l.index(max(curr_probs_l))
    max_location_name = inference.CLASS2NAME[max_location_class]
    max_3_location_classes = sorted(
        range(len(curr_probs_l)), key=lambda sub: curr_probs_l[sub]
    )[-3:]
    max_3_location_classes.reverse()
    max_3_location_names = (
        inference.CLASS2NAME[max_3_location_classes[0]]
        + ","
        + inference.CLASS2NAME[max_3_location_classes[1]]
        + ","
        + inference.CLASS2NAME[max_3_location_classes[2]]
    )
    print(f"Predicted classes: {max_3_location_names}")
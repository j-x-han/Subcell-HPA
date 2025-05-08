import boto3
import csv
import cv2
import numpy as np
from omegaconf import OmegaConf
import os
import utils

import inference
from vit_model import ViTPoolClassifier

def load_antibodies(file_path):
    """
    Load the test antibodies from a file.
    
    Args:
        file_path (str): Path to the file containing the test antibodies.
        
    Returns:
        list: List of test antibodies.
    """
    # This file is copied from the subcell-analysis repo
    print(f"Loading test antibodies from {file_path}...")
    with open(file_path, "r") as f:
        test_antibodies = f.read().splitlines()
    return test_antibodies

def load_metadata(file_path):
    """
    Load the metadata from a CSV file.
    
    Args:
        file_path (str): Path to the metadata CSV file containing cell information.
        
    Returns:
        headers (dict): Dictionary mapping column names to their indices.
        metadata (dict): Dictionary mapping antibody names to their corresponding metadata rows.
    """
    # This file should be downloaded from s3://czi-subcell-public/hpa-processed/cell_crops/metadata.csv
    print(f"Loading metadata from {file_path}...")
    metadata = {}
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Convert headers to a dictionary with column names as keys and indices as values for easier access
        headers = {name: index for index, name in enumerate(headers)}
        for row in reader:
            key = row[headers["antibody"]]
            if key not in metadata:
                metadata[key] = [row]
            else:
                metadata[key].append(row)
    return headers, metadata
    
def preprocess_images(metadata, headers, test_antibodies, s3_bucket, cell_crops_dir, offset, batch_size):
    """
    Preprocess the images by downloading them from S3 and applying the cell mask.
    
    Args:
        metadata (dict): Dictionary mapping antibody names to their corresponding metadata rows.
        headers (dict): Dictionary mapping column names to their indices.
        test_antibodies (list): List of test antibodies.
        s3_bucket (str): S3 bucket name where the images are stored.
        cell_crops_dir (str): Directory to save the cropped images.
        offset (int): Offset for the starting index of the cells to process.
        batch_size (int): Number of cells to process in a batch.
        
    Returns:
        cropped_images (dict): Dictionary mapping file names to their corresponding cropped images.
    """
    # Create the cell crops directory if it doesn't exist
    os.makedirs(cell_crops_dir, exist_ok=True)

    s3 = boto3.client("s3")
    cropped_images = {}
    index = 0
    for antibody in test_antibodies:
        # Process each cell associated with the antibody
        cells = metadata[antibody]
        for cell in cells:
            # Skip cells that are before the offset
            if index < offset:
                index += 1
                continue
            # Return if batch size is reached
            if len(cropped_images) >= batch_size:
                return cropped_images

            if_plate_id = cell[headers["if_plate_id"]]
            position = cell[headers["position"]]
            sample = cell[headers["sample"]]
            cell_id = int(float(cell[headers["cell_id"]]))

            prefix = f"hpa-processed/cell_crops/{if_plate_id}/{if_plate_id}_{position}_{sample}_{cell_id}"
            cell_image_path = f"{prefix}_cell_image.png"
            cell_mask_path = f"{prefix}_cell_mask.png"

            # If the cropped image already exists locally, load it and skip the download
            cropped_image_name = f"{if_plate_id}_{position}_{sample}_{cell_id}_cropped.png"
            cropped_image_path = f"{cell_crops_dir}/{cropped_image_name}"
            if os.path.exists(cropped_image_path):
                print(f"Cropped image {cropped_image_path} already exists, skipping download.")
                cropped_image = cv2.imread(cropped_image_path, cv2.IMREAD_UNCHANGED)
                cropped_images[cropped_image_name] = cropped_image
                continue

            print(f"Loading {cell_image_path} and {cell_mask_path} from S3...")
            cell_image = utils.read_image_from_s3(s3, s3_bucket, cell_image_path)
            cell_mask = utils.read_image_from_s3(s3, s3_bucket, cell_mask_path)

            print(f"Processing {cell_image_path}...")
            # Apply the cell mask to the cell image
            masked_image = cv2.bitwise_and(cell_image, cell_image, mask=cell_mask)

            # Crop the image to 640x640 around the provided x1, x2, y1, y2 coordinates
            x1 = int(float(cell[headers["x1"]]))
            x2 = int(float(cell[headers["x2"]]))
            y1 = int(float(cell[headers["y1"]]))
            y2 = int(float(cell[headers["y2"]]))
            cropped_image = utils.crop_image(masked_image, x1, x2, y1, y2, crop_size=640)

            # Save the cropped image
            cv2.imwrite(cropped_image_path, cropped_image)
            print(f"Saved processed image to {cropped_image_path}")
            cropped_images[cropped_image_name] = cropped_image
    return cropped_images

def load_model(model_config_path, s3_bucket, weights_dir):
    """
    Load the ViTPoolClassifier model and download the weights from S3.
    
    Args:
        model_config_path (str): Path to the model configuration file.
        s3_bucket (str): S3 bucket name where the model weights are stored.
        weights_dir (str): Directory to save the model weights.
        
    Returns:
        model: Loaded ViTPoolClassifier model.
    """
    print("Loading model configuration...")
    with open(model_config_path, "r") as f:
        model_config_file = OmegaConf.load(f)
        config = OmegaConf.to_container(model_config_file)

    classifier_paths = None
    if "classifier_paths" in config:
        classifier_paths = config["classifier_paths"]
    encoder_path = config["encoder_path"]

    # Create the weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)
    # Download the model weights from S3
    s3 = boto3.client("s3")
    encoder_local_path = f"{weights_dir}/{encoder_path.split('/')[-1]}"
    if os.path.exists(encoder_local_path):
        print("Encoder weights already downloaded, skipping download.")
    else:
        print(f"Downloading encoder weights from S3: {encoder_path}")
        utils.download_s3_file(s3, s3_bucket, encoder_path, encoder_local_path)
        print(f"Downloaded encoder weights to {encoder_local_path}")

    classifier_local_paths = []
    for classifier_path in classifier_paths:
        classifier_local_path = f"{weights_dir}/{classifier_path.split('/')[-1]}"
        classifier_local_paths.append(classifier_local_path)
        if os.path.exists(classifier_local_path):
            print(f"Classifier weights {classifier_local_path} already downloaded, skipping download.")
        else:
            print(f"Downloading classifier weights from S3: {classifier_path}")
            utils.download_s3_file(s3, s3_bucket, classifier_path, classifier_local_path)
            print(f"Downloaded classifier weights to {classifier_local_path}")

    print("Loading model...")
    model_config = config.get("model_config")
    model = ViTPoolClassifier(model_config)
    model.load_model_dict(encoder_local_path, classifier_local_paths)
    print("Model loaded successfully.")
    
    return model

def run_inference(model, cropped_image, cropped_image_name, output_dir):
    """
    Run inference on the cropped image using the ViTPoolClassifier model.
    
    Args:
        model (ViTPoolClassifier): Loaded ViTPoolClassifier model.
        cropped_image (numpy.ndarray): Cropped image to run inference on.
        cropped_image_name (str): Name of the cropped image file.
        output_dir (str): Directory to save the output files.
        
    Returns:
        embedding (numpy.ndarray): Embedding from the model.
        probabilities (numpy.ndarray): Probabilities from the model.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Reshape image from (640, 640, 4) -> (4, 640, 640) for the model
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    # Run inference
    print(f"Running inference on {cropped_image_name}...")
    embedding, probabilities = inference.run_model(model, cropped_image, f"{output_dir}/{cropped_image_name}")
    
    return embedding, probabilities

def get_predicted_classes(probabilities):
    """
    Get the predicted classes from the probabilities.
    
    Args:
        probabilities (numpy.ndarray): Probabilities from the model.
        
    Returns:
        str: Predicted classes as a comma-separated string.
    """
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
    return max_3_location_names

def run(
    antibodies_file,
    metadata_file,
    model_config,
    offset,
    batch_size,
    s3_bucket,
    cell_crops_dir,
    weights_dir,
    output_dir
):
    """
    Main function to run the subcell_hpa pipeline.
    
    Args:
        antibodies_file (str): Path to the file containing antibodies of interest.
        metadata_file (str): Path to the metadata file containing cell information.
        model_config (str): Path to the model configuration file.
        batch_size (int): Number of cells to process in a batch.
        s3_bucket (str): S3 bucket name where the images are stored.
        cell_crops_dir (str): Directory to save the cell crops.
        weights_dir (str): Directory to save the model weights.
        output_dir (str): Directory to save the output files.
    """
    # Load test antibodies and metadata
    test_antibodies = load_antibodies(antibodies_file)
    headers, metadata = load_metadata(metadata_file)

    # Preprocess images
    cropped_images = preprocess_images(metadata, headers, test_antibodies, s3_bucket, cell_crops_dir, offset, batch_size)

    # Load the model
    model = load_model(model_config, s3_bucket, weights_dir)

    # Run inference on each cropped image
    for cropped_image_name, cropped_image in cropped_images.items():
        # This will save the embedding, probabilities, and attention map to the output directory
        embedding, probabilities = run_inference(model, cropped_image, cropped_image_name, output_dir)
        
        # Get predicted classes
        predicted_classes = get_predicted_classes(probabilities)
        print(f"Predicted classes: {predicted_classes}")

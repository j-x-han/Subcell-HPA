import boto3
import cv2
import numpy as np
import os

def download_s3_file(s3_client, bucket_name, object_key, local_path):
    """
    Download a file from S3 to a local path.
    
    Args:
        s3_client: Boto3 S3 client.
        bucket_name: Name of the S3 bucket.
        object_key: Key of the object in S3.
        local_path: Local path to save the downloaded file.
    """
    try:
        s3_client.download_file(bucket_name, object_key, local_path)
        print(f"Downloaded {object_key} from s3://{bucket_name} to {local_path}")
    except Exception as e:
        print(f"Failed to download {object_key} from s3://{bucket_name}: {str(e)}")
        raise

def read_image_from_s3(s3_client, bucket_name, object_key):
    """
    Read an image from S3 and decode it into a numpy array.
    
    Args:
        s3_client: Boto3 S3 client.
        bucket_name: Name of the S3 bucket.
        object_key: Key of the object in S3.
    
    Returns:
        Decoded image as a numpy array.
    """
    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    file_content = obj['Body'].read()
    np_arr = np.frombuffer(file_content, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

def crop_image(image, x1, x2, y1, y2, crop_size):
    """
    Crop an image to a specified size around the center of the provided coordinates.
    
    Args:
        image: Input image as a numpy array.
        x1, x2, y1, y2: Coordinates for cropping.
        crop_size: Size of the cropped image (assuming a square crop).
    
    Returns:
        Cropped image as a numpy array.
    """
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    x1 = max(center_x - crop_size // 2, 0)
    x2 = min(center_x + crop_size // 2, image.shape[1])
    y1 = max(center_y - crop_size // 2, 0)
    y2 = min(center_y + crop_size // 2, image.shape[0])
    return image[y1:y2, x1:x2]

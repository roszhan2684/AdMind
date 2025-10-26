import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path: str):
    """
    Reads and resizes the input image to a standard format.
    Returns a normalized numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    
    # Normalize to 0–1 range, but keep float32 precision for TensorFlow
    img_resized = img_resized.astype(np.float32) / 255.0
    return img_resized

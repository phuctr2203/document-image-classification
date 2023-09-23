from __future__ import annotations

import numpy as np
import cv2
import random
from PIL import Image

def rotate_image(image):
    rotation_range = (-30, 30)
    random_angle = random.randint(rotation_range[0], rotation_range[1])

    # Get the size (width, height) of the image
    width, height = image.size

    # Convert the PIL Image to a NumPy array
    image_np = np.array(image)

    # Get the center of the image
    center = (width // 2, height // 2)

    # Perform the rotation on the NumPy array
    rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)

    # Calculate the canvas size to fit the rotated image
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_width = int(height * sin_theta + width * cos_theta)
    new_height = int(height * cos_theta + width * sin_theta)

    # Adjust the translation component of the rotation matrix to center the image on the canvas
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Warp the image to the rotated canvas size with a black background
    rotated_image = cv2.warpAffine(image_np, rotation_matrix, (new_width, new_height), borderValue=(0, 0, 0))

    # Convert the NumPy array back to a PIL Image
    rotated_image_pil = Image.fromarray(rotated_image)

    return rotated_image_pil

def basic_rotate_image(image):
    rotation_range = (-180, 180)
    random_angle = random.randint(rotation_range[0], rotation_range[1])

    # Get the center of the image
    width, height = image.size
    center = (width // 2, height // 2)

    image_np = np.array(image)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)
    rotated_image = cv2.warpAffine(image_np, rotation_matrix, (width, height))

    rotated_image_pil = Image.fromarray(rotated_image)

    return rotated_image_pil
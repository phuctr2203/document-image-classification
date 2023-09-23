from __future__ import annotations

from rotation_augmentation import *
import numpy as np


from PIL import ImageEnhance

def find_colors_to_swap(image):
    flag = True

    # Define a threshold for color similarity (you can adjust this if needed)
    color_similarity_threshold = 50
    min_max_difference_threshold = 40

    # Find pixels that are close to black or white
    black_mask = np.max(image, axis=2) <= color_similarity_threshold
    white_mask = np.min(image, axis=2) >= 255 - color_similarity_threshold

    # Combine the masks to find pixels that are either black or white or similar to them
    close_to_black_or_white_mask = black_mask | white_mask

    min_max_difference_mask = np.max(image, axis=2) - np.min(image, axis=2) >= min_max_difference_threshold

    # Combine the masks to avoid colors with small RGB differences
    not_close_to_black_or_white_mask = ~close_to_black_or_white_mask & min_max_difference_mask

    # Check if there are any distinct colors to swap
    if not np.any(not_close_to_black_or_white_mask):
        #print("Cannot swap colors. Image has only black and white or similar colors.")
        flag = False
        return flag, None

    # Generate a new random color
    new_random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)

    # Create a new image with the same dimensions as the original image
    new_image = np.copy(image)

    # Replace pixels in the new image with the new random color where the original image has colors not close to black or white
    new_image[not_close_to_black_or_white_mask] = new_random_color

    return flag, new_image

def increase_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    enhancement_factor = 10.0
    increased_contrast_image = enhancer.enhance(enhancement_factor)

    new_image = np.copy(increased_contrast_image)

    return new_image
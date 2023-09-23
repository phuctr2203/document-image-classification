from __future__ import annotations

from rotation_augmentation import *
from color_augmentation import *
import numpy as np
import cv2
import os
import easyocr

# Function calculate the size of bbox
def calculate_size(bounding_box):
    # Find bounding box coordinates
    x1, y1 = bounding_box[0]
    x2, y2 = bounding_box[2]

    # Calculate the width and height
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return width, height

# Function check if the bbox satisfy
def sizes_within_tolerance(size1, size2, tolerance):
    width_diff = abs(size1[0] - size2[0])
    height_diff = abs(size1[1] - size2[1])
    return width_diff <= tolerance and height_diff <= tolerance

# Function to prevent coordinate return negative value
def ensure_no_negative(bbox):
    non_negative_bbox = []
    for point in bbox:
        non_negative_point = []
        for coord in point:
            non_negative_coord = max(0, coord)
            non_negative_point.append(non_negative_coord)
        non_negative_bbox.append(non_negative_point)

    return non_negative_bbox

# Function finding the matching bbox
def find_matching_bounding_box(image1, image2, size_tolerance=10):
    reader = easyocr.Reader(['en'])

    # Convert both images into the same color channel
    if image1.shape[2] == 3 and image1.shape[2] != image2.shape[2]:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    elif image2.shape[2] == 3 and image2.shape[2] != image1.shape[2]:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        # Perform OCR on the images
    results1 = reader.readtext(image1, paragraph=True)
    results2 = reader.readtext(image2, paragraph=True)

    # Find matching bounding box sizes within tolerance
    matching_bounding_boxes = []
    for result1 in results1:
        bbox1 = result1[0]
        size1 = calculate_size(bbox1)
        for result2 in results2:
            bbox2 = result2[0]
            size2 = calculate_size(bbox2)
            if sizes_within_tolerance(size1, size2, size_tolerance):
                matching_bounding_boxes.append((bbox1, bbox2))
                break

    return matching_bounding_boxes

# Function swap image based on given bbox
def swap_images_with_bounding_boxes(image1, image2, bbox1, bbox2):
    # Check to ensure not return negative value
    bbox1 = ensure_no_negative(bbox1)
    bbox2 = ensure_no_negative(bbox2)
    
    # Swap the bounding box images
    image1_swapped = image1.copy()
    image1_swapped[bbox1[0][1]:bbox1[2][1], bbox1[0][0]:bbox1[2][0]] = cv2.resize(
        image2[bbox2[0][1]:bbox2[2][1], bbox2[0][0]:bbox2[2][0]],
        (bbox1[2][0] - bbox1[0][0], bbox1[2][1] - bbox1[0][1])
    )

    image2_swapped = image2.copy()
    image2_swapped[bbox2[0][1]:bbox2[2][1], bbox2[0][0]:bbox2[2][0]] = cv2.resize(
        image1[bbox1[0][1]:bbox1[2][1], bbox1[0][0]:bbox1[2][0]],
        (bbox2[2][0] - bbox2[0][0], bbox2[2][1] - bbox2[0][1])
    )

    cv2.rectangle(image1_swapped, (bbox1[0][0], bbox1[0][1]), (bbox1[2][0], bbox1[2][1]), (0, 255, 0), 2)
    cv2.rectangle(image2_swapped, (bbox2[0][0], bbox2[0][1]), (bbox2[2][0], bbox2[2][1]), (0, 255, 0), 2)

    return image1_swapped, image2_swapped


# Apply text augmentation function
def generate_text_augmented_img(image1, image2, size_tolerance=10):
    # Find matching bounding boxes
    matching_bounding_boxes = find_matching_bounding_box(image1, image2, size_tolerance=size_tolerance)

    flag = False

    if not matching_bounding_boxes:
        return flag
    else:
        # Sort the matching bounding boxes based on size
        matching_bounding_boxes.sort(key=lambda bbox: calculate_size(bbox[0]), reverse=True)

        # Select the bounding boxes with the biggest size
        bbox1, bbox2 = matching_bounding_boxes[0]

        # Swap the bounding box images
        image1_swapped, image2_swapped = swap_images_with_bounding_boxes(image1, image2, bbox1, bbox2)

        result_folder = "static/Text_result"
        os.makedirs(result_folder, exist_ok=True)

        # Save the NumPy arrays as image files using OpenCV (cv2)
        result_image_path = os.path.join(result_folder, "result_1.jpg")
        cv2.imwrite(result_image_path, cv2.cvtColor(image1_swapped, cv2.COLOR_BGR2RGB))

        result_image_path = os.path.join(result_folder, "result_2.jpg")
        cv2.imwrite(result_image_path, cv2.cvtColor(image2_swapped, cv2.COLOR_BGR2RGB))

        original_image_path = os.path.join(result_folder, "original_1.jpg")
        cv2.imwrite(original_image_path, cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

        original_image_path = os.path.join(result_folder, "original_2.jpg")
        cv2.imwrite(original_image_path, cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

        flag = True
        return flag
    
import numpy as np

def cutmix_augmentation(image1, image2):
    # Check if the input images have the same dimensions
    alpha=1.0

    if image1.shape != image2.shape:
        # Resize the images to have the same dimensions
        max_height = max(image1.shape[0], image2.shape[0])
        max_width = max(image1.shape[1], image2.shape[1])

        image1 = cv2.resize(image1, (max_width, max_height))
        image2 = cv2.resize(image2, (max_width, max_height))

    # Get image dimensions
    height, width, channels = image1.shape

    # Shuffle the images
    indices = np.arange(height * width)
    np.random.shuffle(indices)

    # Generate random cut parameters
    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1.0 - lam)

    # Calculate the cut coordinates and sizes
    cx = np.random.randint(0, width)
    cy = np.random.randint(0, height)
    w = int(width * cut_ratio)
    h = int(height * cut_ratio)

    # Apply the cut and mix the images
    augmented_image1 = image1.copy()
    augmented_image2 = image2.copy()
    augmented_image1[cy:cy+h, cx:cx+w] = image2[cy:cy+h, cx:cx+w]
    augmented_image2[cy:cy+h, cx:cx+w] = image1[cy:cy+h, cx:cx+w]

    cv2.rectangle(augmented_image1, (cx, cy), (cx + w, cy + h), (0, 255, 0), 2)
    cv2.rectangle(augmented_image2, (cx, cy), (cx + w, cy + h), (0, 255, 0), 2)

    result_folder = "static/Text_result"
    os.makedirs(result_folder, exist_ok=True)

    result_image_path = os.path.join(result_folder, "cutmix_1.jpg")
    cv2.imwrite(result_image_path, cv2.cvtColor(augmented_image1, cv2.COLOR_BGR2RGB))

    result_image_path = os.path.join(result_folder, "cutmix_2.jpg")
    cv2.imwrite(result_image_path, cv2.cvtColor(augmented_image2, cv2.COLOR_BGR2RGB))

    return True
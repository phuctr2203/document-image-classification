"""Inference Module for Document Classification model"""
from __future__ import annotations
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from rotation_augmentation import *
from color_augmentation import *
from text_augmentation import *
import io
import logging
import torch
import fitz
import numpy as np
import os
import time
from typing import List

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from transformers.models import layoutlmv3
from transformers import LayoutLMv3ForSequenceClassification
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, LayoutLMv3Processor

from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()
clients = []
logger = logging.getLogger("Classification Module")

device = torch.device("cpu")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/static", StaticFiles(directory="static"), name="images")

class CertificationType(str, Enum):
    """Certification Type Object"""

    ALLERGEN_STATEMENT = "Allergen_Statement"
    BRC_AUDIT = "BRC_Audit"
    BRC_CERT = "BRC_Certification"
    BIOENGINEER_DOC = "Bioengineer_Document"
    CERT_OF_ANALYSIS = "Certificate_of_Analysis"
    CERT_OF_LIABILITY = "Certificate_of_Liability"
    CERT_OF_ORIGIN = "Certificate_of_Origin"
    FDA_RELATED = "FDA_Related"
    FSSC_AUDIT = "FSSC_Audit"
    FSSC_CERT = "FSSC_Certification"
    GMO_STATEMENT = "GMO_Statement"
    HACCP_PLAN = "HACCP_Plan"
    HALAL = "Halal"
    KOSHER = "Kosher"
    LETTER_OF_GUARANTEE = "Letter_of_Guarantee"
    LOT_CODE = "Lot_Code"
    NUTRITION_SPEC = "Nutrition_Specification"
    ORGANIC = "Organic"
    OTHER = "Other"
    PRODUCT_SPEC = "Product_Specification"
    RECALL_POLICY = "Recall_Policy"
    SQF_AUDIT = "SQF_Audit"
    SQF_CERT = "SQF_Certification"
    SAFETY_DATA_SHEET = "Safety_Data_Sheet"


def convert_to_image(data: bytes):
    with fitz.open(stream=data, filetype="pdf") as doc:
        page_content = doc.load_page(0)
        page_pixmap = page_content.get_pixmap()
        image = page_pixmap.tobytes()

    return image


@lru_cache()
def init_model(
    model_path: Path,
) -> Tuple[
    layoutlmv3.LayoutLMv3ImageProcessor, layoutlmv3.LayoutLMv3ForSequenceClassification
]:
    feature_extractor = LayoutLMv3FeatureExtractor()
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)
    model = LayoutLMv3ForSequenceClassification.from_pretrained("felixtran/lostd_final_model")

    return processor, model


@lru_cache()
@app.post("/inference")
async def inference(data: UploadFile) -> CertificationType:
    """Main function"""
    start_time = time.time()
    processor, model = init_model("")
    image = Image.open(io.BytesIO(convert_to_image(data.file.read()))).convert("RGB")

    encoded_inputs = processor(
        image, return_tensors="pt", padding="max_length", truncation=True
    )

    for k,v in encoded_inputs.items():
        encoded_inputs[k] = v.to(model.device)

    outputs = model(**encoded_inputs)
    loss = outputs.loss
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    end_time = time.time()
    processing_time = end_time - start_time

    result = model.config.id2label[predicted_class_idx]

    return JSONResponse(content={"result": result, "processing_time": processing_time})

@app.post("/rotate-image")
async def rotate_image_api(data: UploadFile):
    # Read the PDF content as bytes
    pdf_content = data.file.read()

    # Extract an image from the PDF (e.g., the first page)
    with fitz.open(stream=pdf_content, filetype="pdf") as pdf_doc:
        pdf_page = pdf_doc.load_page(0)
        pdf_pixmap = pdf_page.get_pixmap()
        pdf_image = Image.frombytes("RGB", [pdf_pixmap.width, pdf_pixmap.height], pdf_pixmap.samples)

    # Perform image rotation
    rotated_image1 = rotate_image(pdf_image)
    rotated_image2 = rotate_image(pdf_image)
    rotated_image3 = rotate_image(pdf_image)
    basic_image = basic_rotate_image(pdf_image)

    result_folder = "static/Rotate_result"
    os.makedirs(result_folder, exist_ok=True)
    result_image_path = os.path.join(result_folder, f"result_1.jpg")
    rotated_image1.save(result_image_path)

    result_image_path = os.path.join(result_folder, f"result_2.jpg")
    rotated_image2.save(result_image_path)

    result_image_path = os.path.join(result_folder, f"result_3.jpg")
    rotated_image3.save(result_image_path)

    original_image_path = os.path.join(result_folder, "original.jpg")
    pdf_image.save(original_image_path, "JPEG")

    result_image_path = os.path.join(result_folder, f"basic.jpg")
    basic_image.save(result_image_path)

    # Return the rotated image as a StreamingResponse with media type "image/jpeg"
    return {"result_image_path": result_image_path}

@app.post("/color-augmentation")
async def color_augmentation_api(data: UploadFile):
    # Read the image content as bytes
    pdf_content = data.file.read()

    # Extract an image from the PDF (e.g., the first page)
    with fitz.open(stream=pdf_content, filetype="pdf") as pdf_doc:
        pdf_page = pdf_doc.load_page(0)
        pdf_pixmap = pdf_page.get_pixmap()
        pdf_image = Image.frombytes("RGB", [pdf_pixmap.width, pdf_pixmap.height], pdf_pixmap.samples)

    # Perform color augmentation
    flag, augmented_image1 = find_colors_to_swap(pdf_image)
    flag, augmented_image2 = find_colors_to_swap(pdf_image)
    flag, augmented_image3 = find_colors_to_swap(pdf_image)
    basic_image = increase_contrast(pdf_image)
    
    if flag == True:
        augmented_image_pil1 = Image.fromarray(augmented_image1)
        augmented_image_pil2 = Image.fromarray(augmented_image2)
        augmented_image_pil3 = Image.fromarray(augmented_image3)
        basic_image_pil = Image.fromarray(basic_image)
        result_folder = "static/Color_result"
        os.makedirs(result_folder, exist_ok=True)
        result_image_path = os.path.join(result_folder, "result_1.jpg")
        augmented_image_pil1.save(result_image_path, "JPEG")

        result_image_path = os.path.join(result_folder, "result_2.jpg")
        augmented_image_pil2.save(result_image_path, "JPEG")

        result_image_path = os.path.join(result_folder, "result_3.jpg")
        augmented_image_pil3.save(result_image_path, "JPEG")

        original_image_path = os.path.join(result_folder, "original.jpg")
        pdf_image.save(original_image_path, "JPEG")

        basic_image_path = os.path.join(result_folder, "basic.jpg")
        basic_image_pil.save(basic_image_path, "JPEG")
        
        return {"success": True}
    else:
        return {"success": False}

@app.post("/text-augmentation")
async def text_augmentation_api(files: List[UploadFile]):
    converted_images = []
    start_time = time.time()

    for file in files:
        pdf_content = await file.read()
        
        # Convert the PDF to an image
        with fitz.open(stream=pdf_content, filetype="pdf") as pdf_doc:
            pdf_page = pdf_doc.load_page(0)
            pdf_pixmap = pdf_page.get_pixmap()
            pdf_image = Image.frombytes("RGB", [pdf_pixmap.width, pdf_pixmap.height], pdf_pixmap.samples)

            pdf_image = np.array(pdf_image)

        # Append the converted image to the list
        converted_images.append(pdf_image)
    
    # Perform color augmentation
    flag = generate_text_augmented_img(converted_images[0], converted_images[1], size_tolerance=10)
    temp = cutmix_augmentation(converted_images[0], converted_images[1])
    
    end_time = time.time()

    processing_time = end_time - start_time
    if flag == True and temp == True:   
        return {"success": True}
    else:
        return {"success": False}

@app.get("/index")
async def serve_html():
    html_path = Path("frontend/index.html")
    return FileResponse(html_path)

@app.get("/model")
async def serve_model_html():
    html_path = Path("frontend/model.html")
    return FileResponse(html_path)

@app.get("/rotate")
async def serve_rotate_html():
    html_path = Path("frontend/rotate.html")
    return FileResponse(html_path)

@app.get("/color")
async def serve_color_html():
    html_path = Path("frontend/color.html")
    return FileResponse(html_path)

@app.get("/text")
async def serve_color_html():
    html_path = Path("frontend/text.html")
    return FileResponse(html_path)
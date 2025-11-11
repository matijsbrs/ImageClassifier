# Inference-API.py
# This script sets up a FastAPI server to handle image uploads for inference.
# It classifies controller device images into four types using a trained MobileNetV2 model
# and extracts the EUI either via OCR (for Type 1) or QR code decoding (for Types 2-4).
# 
# Disclosure: This code is cowritten with AI Tools.
# Author: Matijs Behrens
# Date: 11-11-2025
# Version: 1.0

# Please ensure you have the required libraries installed:
# pip install fastapi uvicorn torch torchvision pillow pyzbar pytesseract python-multipart opencv-python numpy
# sudo apt install python3-pyzbar zbar-tools python3-zbar

# Run the API with:
# .venv/bin/uvicorn src.Inference-API:app --reload --host 0.0.0.0 --port 8000

__version__ = "1.0.0"
__date__    = "2025-11-11"


# 1. Import libraries for inference and API
import io, re
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from pyzbar.pyzbar import decode  # for QR code decoding
import pytesseract                 # for OCR
import os
import cv2


# Redirect stderr to suppress C library warnings from pyzbar
class SuppressStderr:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)
        os.dup2(self.null_fd, 2)
        return self
    
    def __exit__(self, *_):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)
        os.close(self.save_fd)


# 2. Load the trained model weights (ensure model architecture matches training)
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None)  # no pre-trained weights, we will load our own
# Prepare model structure: modify final layer to 4 classes, same as in training
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load("controller_model_weights.pth", map_location=device))
model.eval()  # set to evaluation mode (disable dropout, etc.)
model.to(device)

# Define the same normalization transform as used in training for inference
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3. Utility functions for EUI extraction
def extract_eui_from_text(image: Image.Image) -> str:
    """Extract hex string EUI from an image via OCR (Type 1 devices)."""
    # Convert image to grayscale for better OCR and apply OCR with whitelist for hex characters
    gray = image.convert("L")
    # Configure tesseract to only recognize 0-9 and A-F characters to improve accuracy
    config = "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEF"
    text = pytesseract.image_to_string(gray, config=config)
    # Use regex to find a hex string in the OCR result
    match = re.search(r'[0-9A-F]+', text.upper())
    return ((match.group(0), "No action needed" )if match else ("", "action: verify OCR accuracy or improve image quality"))  # return the found hex string (or empty if not found)

def extract_eui_from_qr_original(image: Image.Image) -> str:
    """Extract EUI from a QR code in the image (Types 2-4 devices)."""
    # Convert PIL image to NumPy array for pyzbar
    img_array = np.array(image.convert("RGB"))
    results = decode(img_array)
    if results:
        # Assume first decoded QR contains the EUI
        qr_data = results[0].data.decode("utf-8")
        return qr_data.strip()
    else:
        return ""  # no QR code found or decoding failed
    
def extract_eui_from_qr(image: Image.Image):
    try:
        # Convert PIL Image to OpenCV format for QR/Barcode detection
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
        
        # Strategy 1: Try direct detection on original image
        with SuppressStderr():
            decoded_objects = decode(image_cv)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (original): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # Convert to grayscale for all subsequent operations
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv
        
        # Strategy 2: Try INVERTED image (for white QR codes on black background)
        print("[QR] Original failed, trying inverted image...")
        inverted = cv2.bitwise_not(gray)
        with SuppressStderr():
            decoded_objects = decode(inverted)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (inverted): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # Strategy 3: Otsu's thresholding (auto-determines optimal threshold)
        print("[QR] Inverted failed, trying Otsu's threshold...")
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        with SuppressStderr():
            decoded_objects = decode(otsu)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (Otsu): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # Strategy 4: Otsu's threshold INVERTED
        print("[QR] Otsu failed, trying inverted Otsu...")
        otsu_inv = cv2.bitwise_not(otsu)
        with SuppressStderr():
            decoded_objects = decode(otsu_inv)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (Otsu inverted): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # Strategy 5: Morphological operations to clean up noise
        print("[QR] Otsu inverted failed, trying morphological cleanup...")
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        with SuppressStderr():
            decoded_objects = decode(morph)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (morphological): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # Strategy 6: Enhance contrast with CLAHE then invert
        print("[QR] Morphological failed, trying CLAHE + invert...")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced_inv = cv2.bitwise_not(enhanced)
        with SuppressStderr():
            decoded_objects = decode(enhanced_inv)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (CLAHE inverted): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # Strategy 7: Sharpen + threshold
        print("[QR] CLAHE failed, trying sharpen + threshold...")
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        with SuppressStderr():
            decoded_objects = decode(sharp_thresh)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (sharpened): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # Strategy 8: Sharpen + threshold INVERTED
        sharp_thresh_inv = cv2.bitwise_not(sharp_thresh)
        with SuppressStderr():
            decoded_objects = decode(sharp_thresh_inv)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (sharpened inverted): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # Strategy 9: Adaptive thresholding (both regular and inverted)
        print("[QR] Sharpened failed, trying adaptive threshold...")
        for block_size in [11, 15, 21, 31]:
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
            )
            with SuppressStderr():
                decoded_objects = decode(adaptive_thresh)
            if decoded_objects:
                detected_text = decoded_objects[0].data.decode('utf-8')
                print(f"[QR/Barcode] Detected (adaptive {block_size}): '{detected_text}'")
                return detected_text, "no further action needed"

            
            # Try inverted too
            adaptive_inv = cv2.bitwise_not(adaptive_thresh)
            with SuppressStderr():
                decoded_objects = decode(adaptive_inv)
            if decoded_objects:
                detected_text = decoded_objects[0].data.decode('utf-8')
                print(f"[QR/Barcode] Detected (adaptive {block_size} inverted): '{detected_text}'")
                return detected_text, "no further action needed"

    
        
        # Strategy 10: Bilateral filter (preserves edges while reducing noise)
        print("[QR] Scaling failed, trying bilateral filter...")
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        with SuppressStderr():
            decoded_objects = decode(bilateral_thresh)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (bilateral): '{detected_text}'")
            return detected_text, "no further action needed"
        
        bilateral_inv = cv2.bitwise_not(bilateral_thresh)
        with SuppressStderr():
            decoded_objects = decode(bilateral_inv)
        if decoded_objects:
            detected_text = decoded_objects[0].data.decode('utf-8')
            print(f"[QR/Barcode] Detected (bilateral inverted): '{detected_text}'")
            return detected_text, "no further action needed"
        
        # No QR code found after all strategies
        return "Not found", "action: try better lighting, different distance, or another camera angle."
        
    except Exception as e:
        print(f"[QR Scan Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "action: check server logs for details."

# 4. Prediction function that uses the model to classify and then extract EUI
def classify_and_extract(image: Image.Image):
    # Apply the same transforms as training
    img_tensor = infer_transform(image).unsqueeze(0)  # add batch dimension
    img_tensor = img_tensor.to(device)
    # Run model inference
    with torch.no_grad():  # no grad for efficiency
        outputs = model(img_tensor)
        # Get predicted class index
        pred_idx = int(torch.argmax(outputs, dim=1).item())
    
    # Extract EUI using appropriate method
    if pred_idx == 0:
        eui, action = extract_eui_from_text(image)
        action = "action: verify OCR accuracy or improve image quality"
        if len(eui) == 8:
            if eui.startswith("1441"):
                action = "No action needed"
            else:
                eui = ""
        else:
            eui = ""
        
    else:
        eui, action = extract_eui_from_qr(image)
        if eui.lower().startswith("70b3d5b02013") or eui.lower().startswith("70b3d5b02014"):
            pred_idx = 2  # Nordic Automation Systems UL2030-UL2033
        elif eui.lower().startswith("70b3d5b02015"):
            pred_idx = 3  # Nordic Automation Systems UL2034    

    # Map class index to type and brand
    label_map = {
        0: {"type": "Telecontroller", "brand": "Ziut"},                           # Type 1
        1: {"type": "RMC-PUK", "brand": "Remoticom"},                             # Type 2
        2: {"type": "UL2030-UL2033", "brand": "Nordic Automation Systems"},       # Type 3
        3: {"type": "UL2034", "brand": "Nordic Automation Systems"}               # Type 4
    }
    result = label_map.get(pred_idx, {"type": "Unknown", "brand": "Unknown"})
    
    result["EUI"] = eui
    result["action"] = action
    return result

# 5. Set up FastAPI app
app = FastAPI()

# Define an endpoint for predictions
@app.post("/predict/")
async def predict_controller(file: UploadFile = File(...)):
    # Read image data from the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    # Ensure the image is in a suitable mode (convert to RGB)
    image = image.convert("RGB")
    # Run classification and EUI extraction
    result = classify_and_extract(image)
    # Include the filename in the result
    result["filename"] = file.filename
    return result

# Inference-API.py
# This script sets up a FastAPI server to handle image uploads for inference.
# It classifies controller device images into types using a trained MobileNetV2 model
# and returns metadata defined in the training configuration.
# 
# Disclosure: This code is cowritten with AI Tools.
# Author: Matijs Behrens
# Date: 11-11-2025
# Version: 1.1

# Please ensure you have the required libraries installed:
# pip install fastapi uvicorn torch torchvision pillow python-multipart

# Run the API with:
# .venv/bin/uvicorn src.Inference-API:app --reload --host 0.0.0.0 --port 8000

__version__ = "1.0.0"
__date__    = "2025-11-11"


# 1. Import libraries for inference and API
from datetime import datetime
import io, json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
from torchvision import models, transforms
import os
from pathlib import Path


# 2. Load the trained model weights (ensure model architecture matches training)
device = torch.device("cpu")
CLASS_NAMES_PATH = Path("class_names.json")
try:
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as class_file:
        CLASS_NAMES = json.load(class_file)
except FileNotFoundError:
    CLASS_NAMES = []
    

model = models.mobilenet_v2(weights=None)  # no pre-trained weights, we will load our own
# Prepare model structure: modify final layer to match number of classes from training
num_ftrs = model.classifier[1].in_features
num_model_classes = len(CLASS_NAMES) if CLASS_NAMES else 4
model.classifier[1] = torch.nn.Linear(num_ftrs, num_model_classes)
model.load_state_dict(torch.load("controller_model_weights.pth", map_location=device))
model.eval()  # set to evaluation mode (disable dropout, etc.)
model.to(device)


num_model_classes = model.classifier[1].out_features
if not CLASS_NAMES or len(CLASS_NAMES) != num_model_classes:
    CLASS_NAMES = [f"type{i}" for i in range(num_model_classes)]

# Load class metadata (generated during training)
CLASS_INFO_PATH = Path("class_info.json")
try:
    with CLASS_INFO_PATH.open("r", encoding="utf-8") as info_file:
        CLASS_INFO = json.load(info_file)
except FileNotFoundError:
    print("Warning: class_info.json not found. Using empty metadata.")
    CLASS_INFO = {}

# Define the same normalization transform as used in training for inference
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])




# 4. Prediction function that uses the model to classify and then extract EUI
def classify_and_extract(image: Image.Image):
    # Apply the same transforms as training
    img_tensor = infer_transform(image).unsqueeze(0)  # add batch dimension
    img_tensor = img_tensor.to(device)
    # Run model inference
    with torch.no_grad():  # no grad for efficiency
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        # Get predicted class index
        pred_idx = int(torch.argmax(probabilities, dim=1).item())

    pred_class = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else str(pred_idx)
    confidence = float(probabilities[0, pred_idx].item())

    # Retrieve metadata for the predicted class
    details = CLASS_INFO.get(pred_class, {}).copy()
    
    # If metadata is missing, provide defaults
    if not details:
        details = {
            "description": "Unknown",
            "found": "No",
            "connection": "Unknown",
            "brand": "Unknown",
            "class": "Unknown",
            "type": "Unknown"
        }

    # Add confidence score
    details["Confidence"] = f"{confidence:.0%}"
    
    return details

# 5. Set up FastAPI app
app = FastAPI()

# 6. Add a root endpoint for proof of concept
@app.get("/", response_class=HTMLResponse)
async def root():
    """Return a simple web page for testing the image classification API."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Classifier - Controller Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .upload-section {
                margin: 20px 0;
                padding: 20px;
                border: 2px dashed #ccc;
                border-radius: 5px;
                text-align: center;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            #preview {
                margin: 20px 0;
                text-align: center;
            }
            #preview img {
                max-width: 100%;
                max-height: 400px;
                border-radius: 5px;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
                display: none;
            }
            .result-item {
                margin: 8px 0;
                padding: 8px;
                background-color: white;
                border-left: 4px solid #4CAF50;
            }
            .result-label {
                font-weight: bold;
                color: #666;
            }
            .error {
                color: #d32f2f;
                background-color: #ffebee;
                border-left-color: #d32f2f;
            }
            .loading {
                text-align: center;
                color: #666;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Controller Image Classifier</h1>
            <p style="text-align: center; color: #666;">Upload an image to classify controller devices</p>
            
            <div class="upload-section">
                <input type="file" id="imageInput" accept="image/*" />
                <br>
                <button id="uploadBtn" onclick="uploadImage()">Classify Image</button>
            </div>
            
            <div id="preview"></div>
            <div class="loading" id="loading">Processing image...</div>
            <div id="result"></div>
        </div>

        <script>
            const imageInput = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            const uploadBtn = document.getElementById('uploadBtn');
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');

            imageInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.innerHTML = '<img src="' + e.target.result + '" alt="Preview">';
                    };
                    reader.readAsDataURL(file);
                    uploadBtn.disabled = false;
                } else {
                    preview.innerHTML = '';
                    uploadBtn.disabled = true;
                }
                resultDiv.style.display = 'none';
            });

            async function uploadImage() {
                const file = imageInput.files[0];
                if (!file) {
                    alert('Please select an image first');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                uploadBtn.disabled = true;
                loadingDiv.style.display = 'block';
                resultDiv.style.display = 'none';

                try {
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    displayResult(data);
                } catch (error) {
                    displayError('Error: ' + error.message);
                } finally {
                    uploadBtn.disabled = false;
                    loadingDiv.style.display = 'none';
                }
            }

            function displayResult(data) {
                let html = '<h3>Classification Results:</h3>';
                
                for (const [key, value] of Object.entries(data)) {
                    html += '<div class="result-item">';
                    html += '<span class="result-label">' + capitalize(key) + ':</span> ';
                    html += '<span>' + value + '</span>';
                    html += '</div>';
                }

                resultDiv.innerHTML = html;
                resultDiv.style.display = 'block';
            }

            function displayError(message) {
                resultDiv.innerHTML = '<div class="result-item error">' + message + '</div>';
                resultDiv.style.display = 'block';
            }

            function capitalize(str) {
                return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            }

            // Disable button initially
            uploadBtn.disabled = true;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Define an endpoint for predictions
@app.post("/predict/")
async def predict_controller(file: UploadFile = File(...)):
    start_time = datetime.now()
    
    # Read image data from the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    # Ensure the image is in a suitable mode (convert to RGB)
    image = image.convert("RGB")
    # Run classification and EUI extraction
    result = classify_and_extract(image)
    # Include the filename in the result
    result["filename"] = file.filename
    result["processing_duration_ms"] = (int((datetime.now() - start_time).total_seconds() * 1000))

    return result



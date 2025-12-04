import os
import json
import base64
import re
import csv
import shutil
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Configuration
OLLAMA_HOST = "http://192.168.2.64:11434"
INPUT_FOLDER = "./"  # Change this to your folder path
OUTPUT_HTML = "./ocr_results.html"
OUTPUT_CSV = "./ocr_results.csv"
PROCESSED_FOLDER = "./processed"

# Model configurations - models are tried in order until one succeeds
MODELS = [
    { # First model configuration This model is lightweight and can run on the GPU
        "name": "qwen2.5vl:3b",
        "prompt": "Spot all the text in the image with word-level and output in JSON format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...].",
        "options": {
            "temperature": 0.0001,
            "max_temperature": 2.1,
            "step_temperature": 0.5
        }
    },
    { # Second model configuration This model is more memory intensive and cannot run on the GPU, but yields better results for the cost of speed/time
        "name": "qwen3-vl:2b-instruct",
        "prompt": "Spot all the text in the image with word-level and output in JSON format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...].",
        "options": {
            "temperature": 0.0001,
            "max_temperature": 2.1,
            "step_temperature": 0.5
        }
    },
    { # Third model configuration This model is more memory intensive and cannot run on the GPU, It has a different internal structure than the previous two models
      # but may yield better results for certain images.
        "name": "ministral-3:3b-instruct-2512-q4_K_M",
        "prompt": "Read the text in the image and output in full words! in JSON format as [{'text_content': 'text'}, ...].",
        "options": {
            "temperature": 0.0001,
            "max_temperature": 1.51,
            "step_temperature": 0.5
        }
    }
]


def load_existing_csv(csv_path):
    """Load existing CSV and return sets of processed files, found hex numbers, and failed files"""
    processed_files = set()
    found_hex_numbers = set()
    failed_files = set()
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('filename', '')
                    status = row.get('status', 'success')
                    
                    # Failed files should be retried, so don't add to processed
                    if status == 'failed':
                        failed_files.add(filename)
                    else:
                        processed_files.add(filename)
                        hex_num = row.get('hex_text', '').strip().upper()
                        if hex_num:
                            found_hex_numbers.add(hex_num)
            print(f"Loaded {len(processed_files)} processed files, {len(found_hex_numbers)} hex numbers, {len(failed_files)} failed files from CSV")
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    return processed_files, found_hex_numbers, failed_files


def ensure_csv_header(csv_path):
    """Ensure CSV file exists with header"""
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['filename', 'status', 'hex_text', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"Created new CSV file: {csv_path}")


def append_to_csv(csv_path, filename, detections, existing_hex_numbers, status='success'):
    """Append detections for a single image to CSV immediately"""
    new_entries = 0
    skipped_duplicates = 0
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'status', 'hex_text', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        for det in detections:
            hex_text = det.get('text_content', '').strip().upper()
            
            # Skip if hex number already exists
            if hex_text in existing_hex_numbers:
                skipped_duplicates += 1
                print(f"   Skipping duplicate hex number: {hex_text}")
                continue
            
            bbox = det.get('bbox_2d', [0, 0, 0, 0])
            writer.writerow({
                'filename': filename,
                'status': status,
                'hex_text': hex_text,
                'bbox_x1': bbox[0] if len(bbox) > 0 else 0,
                'bbox_y1': bbox[1] if len(bbox) > 1 else 0,
                'bbox_x2': bbox[2] if len(bbox) > 2 else 0,
                'bbox_y2': bbox[3] if len(bbox) > 3 else 0
            })
            existing_hex_numbers.add(hex_text)
            new_entries += 1
    
    if new_entries > 0 or skipped_duplicates > 0:
        print(f"   CSV: {new_entries} new, {skipped_duplicates} duplicates skipped")
    
    return new_entries


def append_failure_to_csv(csv_path, filename):
    """Append a failed image entry to CSV"""
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'status', 'hex_text', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            'filename': filename,
            'status': 'failed',
            'hex_text': '',
            'bbox_x1': 0,
            'bbox_y1': 0,
            'bbox_x2': 0,
            'bbox_y2': 0
        })
    print(f"   CSV: Recorded failure for {filename}")


def copy_to_processed(source_path, hex_text, processed_folder=PROCESSED_FOLDER):
    """
    Copy file to processed folder with new naming convention.
    
    Args:
        source_path: Path to the source image file
        hex_text: The hex text to include in filename (or 'Failed' for failed files)
        processed_folder: Destination folder path
    
    Returns:
        Path to the copied file or None on error
    """
    try:
        # Ensure processed folder exists
        os.makedirs(processed_folder, exist_ok=True)
        
        source = Path(source_path)
        original_name = source.stem  # filename without extension
        extension = source.suffix  # .jpg, .png, etc.
        
        # Create new filename: originalname.hex_text.extension
        new_filename = f"{original_name}.{hex_text}{extension}"
        dest_path = Path(processed_folder) / new_filename
        
        # Copy the file
        shutil.copy2(source_path, dest_path)
        print(f"   Copied to: {dest_path}")
        
        return dest_path
    except Exception as e:
        print(f"   Error copying file to processed folder: {e}")
        return None

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_loop_in_response(response_text, min_pattern_length=35, min_repetitions=20):
    """
    Detect if the LLM got stuck in a loop by checking for repeated patterns.
    
    Args:
        response_text: The response text to check
        min_pattern_length: Minimum length of pattern to consider (to avoid false positives)
        min_repetitions: Minimum number of repetitions to consider it a loop
    
    Returns:
        tuple: (is_loop_detected, pattern_found, repetition_count)
    """
    if not response_text or len(response_text) < min_pattern_length * min_repetitions:
        return False, None, 0
    
    # Check for repeated substrings of various lengths
    for pattern_len in range(min_pattern_length, len(response_text) // min_repetitions + 1):
        # Try patterns starting from different positions
        for start in range(min(100, len(response_text) - pattern_len * min_repetitions)):
            pattern = response_text[start:start + pattern_len]
            
            # Skip patterns that are mostly whitespace
            if len(pattern.strip()) < min_pattern_length // 2:
                continue
            
            # Count occurrences
            count = response_text.count(pattern)
            
            if count >= min_repetitions:
                # Verify it's actually consecutive/nearby repetitions (not just common words)
                # Check if the pattern appears in a concentrated area
                first_pos = response_text.find(pattern)
                expected_end = first_pos + (pattern_len * count) + (count * 10)  # Allow some slack
                
                if expected_end < len(response_text) * 1.5:  # Repetitions are clustered
                    return True, pattern[:50] + "..." if len(pattern) > 50 else pattern, count
    
    # Also check for very long responses which might indicate runaway generation
    if len(response_text) > 10000:
        # Check if the response has too many similar JSON objects
        json_object_count = response_text.count('"bbox_2d"')
        if json_object_count > 50:  # Unreasonably many detections
            return True, f"Excessive JSON objects ({json_object_count})", json_object_count
    
    return False, None, 0

def process_image_with_model(image_path, model_config, temperature):
    """
    Send image to Ollama using the specified model configuration.
    
    Args:
        image_path: Path to the image file
        model_config: Dictionary with model name, prompt, and options
        temperature: Temperature to use for this attempt
    
    Returns:
        tuple: (response_text, status) where status is 'success', 'loop', or 'error'
    """
    base64_image = encode_image_to_base64(image_path)
    
    payload = {
        "model": model_config["name"],
        "prompt": model_config["prompt"],
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": temperature
        },
    }
    
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        # Check for loop detection
        is_loop, pattern, count = detect_loop_in_response(response_text)
        if is_loop:
            print(f"⚠️  Loop detected in response!")
            print(f"   Pattern repeated {count} times: {pattern}")
            return None, "loop"
        
        return response_text, "success"
    except Exception as e:
        print(f"   Error: {e}")
        return None, "error"


def process_image_with_retry(image_path):
    """
    Process image by trying each model in the MODELS list.
    For each model, retry with increasing temperature on failure.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Response text or None if all models and attempts fail
    """
    for model_index, model_config in enumerate(MODELS):
        model_name = model_config["name"]
        options = model_config["options"]
        
        start_temp = options.get("temperature", 0.0001)
        max_temp = options.get("max_temperature", 1.5)
        step = options.get("step_temperature", 0.25)
        
        print(f"   [{model_index + 1}/{len(MODELS)}] Trying model: {model_name}")
        
        temperature = start_temp
        attempt = 1
        
        while temperature <= max_temp:
            print(f"      Attempt {attempt} with temperature={temperature:.4f}")
            
            response, status = process_image_with_model(image_path, model_config, temperature)
            
            if status == "success" and response:
                try:
                    hex_detections = [
                        det for det in extract_json_from_response(response) 
                        if is_controller_id(det.get('text_content', ''))
                    ]
                    if hex_detections and len(hex_detections) > 0:
                        return hex_detections
                    else:
                        print(f"json extraction successful but no valid hex detections found.")
                        continue
                except:
                    continue # Try next attempt
                
            
            temperature += step
            attempt += 1
            
            if temperature <= max_temp:
                if status == "loop":
                    print(f"      Retrying with higher temperature...")
                elif status == "error":
                    print(f"      Retrying with higher temperature...")
                else:
                    print(f"      Empty response, retrying with higher temperature...")
        
        print(f"   ❌ Model {model_name} failed after {attempt-1} attempts")
        
        if model_index < len(MODELS) - 1:
            print(f"   Trying next model...")
    
    print(f"   ❌ All {len(MODELS)} models failed for {image_path}")
    return None

def extract_json_from_response(response_text):
    """Extract JSON array from response text"""
    try:
        # Try to find JSON array in the response
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            # Remove trailing commas before ] or } (invalid JSON but common in LLM output)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r'\'', '\"', json_str)
            print(f"Extracted JSON: {json_str}")
            return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Failed on: Response text was: {response_text}")
    return None

def is_controller_id(text):
    """
    This function checks if the provided text is a valid hexadecimal string. following the pattern of controller IDs.
    A controller ID consists of 8 hexadecimal characters (0-9, A-F).
    """
    pattern = r'^[0-9A-Fa-f]{8}$'
    return re.match(pattern, text.strip()) is not None
    
def draw_bboxes_on_image(image_path, detections):
    """Draw bounding boxes on image and return base64 encoded result"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for det in detections:
        bbox = det.get('bbox_2d', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    # Save to bytes
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def process_folder(folder_path, csv_path, processed_files, existing_hex_numbers, failed_files):
    """Process all images in folder, saving to CSV after each image"""
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    # Ensure CSV file exists with header
    ensure_csv_header(csv_path)
    
    results = []
    failed_results = []
    skipped_files = 0
    total_new_entries = 0
    
    for image_file in folder.iterdir():
        if image_file.suffix.lower() in image_extensions:
            # Skip already processed files (but not failed ones - they should be retried)
            if image_file.name in processed_files:
                print(f"Skipping already processed: {image_file.name}")
                skipped_files += 1
                continue
            
            # Note if this was previously failed
            is_retry = image_file.name in failed_files
            if is_retry:
                print(f"Retrying previously failed: {image_file.name}...")
            else:
                print(f"Processing {image_file.name}...")
            
            # Get OCR results from Ollama with retry logic
            response = process_image_with_retry(str(image_file))
            if response is None or len(response) == 0:
                # Record failure to CSV
                append_failure_to_csv(csv_path, image_file.name)
                # Copy failed file to processed folder with 'Failed' label
                copy_to_processed(str(image_file), 'Failed')
                failed_results.append({
                    'filename': image_file.name,
                    'image_path': str(image_file)
                })
                continue
            
            # Extract JSON and filter for hexadecimal strings
            # detections = extract_json_from_response(response)
            
            
            
            else:
                # Filter out already known hex numbers for display purposes
                new_hex_detections = [
                    det for det in response
                    if det.get('text_content', '').strip().upper() not in existing_hex_numbers
                ]
                
                # if new_hex_detections:
                # Save to CSV immediately after processing each image
                new_entries = append_to_csv(csv_path, image_file.name, new_hex_detections, existing_hex_numbers)
                total_new_entries += new_entries
                
                # Copy file to processed folder with first hex_text in filename
                first_hex = response[0].get('text_content', '').strip().upper()
                copy_to_processed(str(image_file), first_hex)
                
                # Add to processed files set
                processed_files.add(image_file.name)
                
                # Draw bounding boxes on image for HTML
                img_with_boxes = draw_bboxes_on_image(str(image_file), new_hex_detections)
                
                results.append({
                    'filename': image_file.name,
                    'image_with_boxes': img_with_boxes,
                    'detections': new_hex_detections
                })
                # else:
                #     print(f"   All hex numbers already known, skipping...")
    
    if skipped_files > 0:
        print(f"\nSkipped {skipped_files} already processed files")
    
    if failed_results:
        print(f"Failed to process {len(failed_results)} images")
    
    print(f"Total new CSV entries: {total_new_entries}")
    
    return results, failed_results

def generate_html(results, failed_results, output_path):
    """Generate HTML file with results including failed files"""
    success_count = len(results)
    failed_count = len(failed_results)
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR Hexadecimal Detection Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        th.failed {{
            background-color: #d32f2f;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        img {{
            max-width: 600px;
            height: auto;
            border: 1px solid #ddd;
        }}
        img.failed-img {{
            max-width: 300px;
            border: 2px solid #d32f2f;
        }}
        .detection-item {{
            margin: 5px 0;
            padding: 5px;
            background-color: #f9f9f9;
            border-left: 3px solid #4CAF50;
        }}
        .hex-text {{
            font-family: 'Courier New', monospace;
            font-weight: bold;
            color: #d32f2f;
        }}
        .failed-section {{
            margin-top: 40px;
        }}
        .failed-item {{
            display: inline-block;
            margin: 10px;
            padding: 10px;
            background-color: #fff;
            border: 2px solid #d32f2f;
            border-radius: 5px;
        }}
        .failed-label {{
            color: #d32f2f;
            font-weight: bold;
            margin-top: 5px;
        }}
        .summary {{
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>OCR Hexadecimal Detection Results</h1>
    <div class="summary">
        <strong>Summary:</strong> {success_count} successful, {failed_count} failed
    </div>
"""
    
    if results:
        html += """
    <h2>✅ Successfully Processed Images</h2>
    <table>
        <tr>
            <th>Image with Bounding Boxes</th>
            <th>Detected Hexadecimal Strings</th>
        </tr>
"""
        
        for result in results:
            html += f"""
        <tr>
            <td>
                <h3>{result['filename']}</h3>
                <img src="data:image/png;base64,{result['image_with_boxes']}" alt="{result['filename']}">
            </td>
            <td>
"""
            for det in result['detections']:
                text = det.get('text_content', '')
                bbox = det.get('bbox_2d', [])
                html += f"""
                <div class="detection-item">
                    <span class="hex-text">{text}</span><br>
                    <small>BBox: {bbox}</small>
                </div>
"""
            html += """
            </td>
        </tr>
"""
        
        html += """
    </table>
"""
    
    if failed_results:
        html += """
    <div class="failed-section">
        <h2>❌ Failed Images (Will Be Retried)</h2>
        <table>
            <tr>
                <th class="failed">Image</th>
                <th class="failed">Filename</th>
            </tr>
"""
        
        for failed in failed_results:
            # Encode the failed image for display
            try:
                img_base64 = encode_image_to_base64(failed['image_path'])
                html += f"""
            <tr>
                <td>
                    <img class="failed-img" src="data:image/png;base64,{img_base64}" alt="{failed['filename']}">
                </td>
                <td>
                    <span class="failed-label">{failed['filename']}</span><br>
                    <small>Processing failed after all retry attempts</small>
                </td>
            </tr>
"""
            except:
                html += f"""
            <tr>
                <td><em>Image could not be loaded</em></td>
                <td>
                    <span class="failed-label">{failed['filename']}</span><br>
                    <small>Processing failed after all retry attempts</small>
                </td>
            </tr>
"""
        
        html += """
        </table>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nHTML report generated: {output_path}")

def main():
    print("Starting OCR processing...")
    print(f"Folder: {INPUT_FOLDER}")
    print(f"Ollama Host: {OLLAMA_HOST}")
    print(f"Models configured: {len(MODELS)}")
    for i, model in enumerate(MODELS):
        print(f"  [{i+1}] {model['name']} (temp: {model['options']['temperature']} -> {model['options']['max_temperature']})")
    print(f"CSV Output: {OUTPUT_CSV}\n")
    
    # Load existing data from CSV
    processed_files, existing_hex_numbers, failed_files = load_existing_csv(OUTPUT_CSV)
    
    # Process folder - CSV is written after each image
    results, failed_results = process_folder(INPUT_FOLDER, OUTPUT_CSV, processed_files, existing_hex_numbers, failed_files)
    
    if results or failed_results:
        # Generate HTML report at the end
        generate_html(results, failed_results, OUTPUT_HTML)
        print(f"\nProcessed {len(results)} images with new hexadecimal content.")
        if failed_results:
            print(f"Failed to process {len(failed_results)} images (recorded in CSV for retry).")
    else:
        print("\nNo new hexadecimal strings detected in any images.")

if __name__ == "__main__":
    main()

import os
import glob
import pydicom
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

def crop_breast_roi(img):
    """
    Ian Pan's deterministic OpenCV cropping logic.
    Strips away background padding, noise, and labels to isolate the tissue.
    """
    margin = 5
    if img.shape[0] <= margin * 2 or img.shape[1] <= margin * 2:
        return img
    
    # 1. Slice off outer edges to ignore potential border artifacts
    img_inner = img[margin:-margin, margin:-margin]
    
    # PATCH 1: Dynamic thresholding based on image intensity to support float matrix
    threshold = np.max(img_inner) * 0.05
    binary_mask = (img_inner > threshold).astype("uint8")
    
    # 2. Component analysis to isolate the biggest continuous structure
    output = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
    stats = output[2]
    
    if len(stats) <= 1:
        return img  # Fallback if no clean components are isolated
        
    # Index 0 is background; locate largest actual object component
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    
    # 3. Map coordinates back to original scale with a minor protective buffer
    x1 = max(0, x1 + margin - 5)
    y1 = max(0, y1 + margin - 5)
    x2 = min(img.shape[1], x1 + w + 10)
    y2 = min(img.shape[0], y1 + h + 10)
    
    return img[y1:y2, x1:x2]

def load_mammo_model(device):
    print("\n[1/4] Initializing Ian Pan's Clinical Architecture (mammoscreen)...")
    try:
        print("      -> Pulling/Loading weights from Hugging Face Hub...")
        # Note: If running offline in the van, add 'local_files_only=True' below
        model = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed to load model from Hugging Face: {e}")
        print("Make sure you have an active internet connection for the FIRST run.")
        exit(1)

def process_single_image():
    # Find a DICOM file in the input directory
    dcm_files = glob.glob(os.path.join(INPUT_DIR, '*.dcm'))
    
    if not dcm_files:
        print(f"\n[ERROR] No .dcm files found in {INPUT_DIR}")
        print("Please place a DICOM file in the input folder and try again.")
        return

    file_path = dcm_files[0]
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    
    print(f"\n[2/4] Reading & Cropping DICOM Image: {filename}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mammo_model(device)
    
    # 1. Load Image using the model's built-in DICOM parser
    try:
        img_array = model.load_image_from_dicom(file_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to parse DICOM pixel array: {e}")
        exit(1)
        
    # PATCH 3: Safe normalization of float matrices to 8-bit integers before cropping and saving
    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    # 2. Extract metadata using pydicom for reporting
    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
    patient_id = getattr(ds, 'PatientID', 'Unknown_ID')
    view_pos = getattr(ds, 'ViewPosition', 'CC').upper()  # Default to CC if missing
    
    # 3. Apply OpenCV Cropping Optimization before Inference
    print("      -> Optimizing image: Cropping dead space and text burns...")
    img_array = crop_breast_roi(img_array)
    
    # Save the optimized PNG version for checking/dashboard
    png_img = Image.fromarray(img_array)
    png_filename = f"{base_name}_{patient_id}_{view_pos}.png"
    png_filepath = os.path.join(OUTPUT_DIR, png_filename)
    png_img.save(png_filepath)
    print(f"      -> Converted and saved cropped PNG: {png_filename}")

    # 4. Run Inference
    print("\n[3/4] Running AI Inference (Cancer Risk & Density)...")
    
    # PATCH 2 (Updated): Duplicate the available view to satisfy the ensemble math
    # and prevent 'NoneType' copy crashes inside the Hugging Face module.
    breast_dict = {}
    if 'MLO' in view_pos:
        breast_dict['mlo'] = img_array
        breast_dict['cc'] = img_array  # Duplicate missing view
    else:
        breast_dict['cc'] = img_array
        breast_dict['mlo'] = img_array # Duplicate missing view
        
    with torch.inference_mode():
        # Pass as a list of dictionaries (Batch size of 1 patient)
        output = model([breast_dict], device=device)
        
    # Extract Cancer Probability
    cancer_prob = output['cancer'][0].item() * 100
    cancer_class = "Suspicious/Malignant" if cancer_prob > 50 else "Negative/Benign"
    
    # Extract Breast Density (ACR Categories A-D)
    density_idx = output['density'][0].argmax(0).item()
    density_classes = [
        "A (Almost entirely fatty)", 
        "B (Scattered fibroglandular)", 
        "C (Heterogeneously dense)", 
        "D (Extremely dense)"
    ]
    density_class = density_classes[density_idx]
    
    print(f"      -> Cancer Risk: {cancer_class} ({cancer_prob:.1f}%)")
    print(f"      -> Breast Density: {density_class}")

    # 5. Save Text Report
    print("\n[4/4] Saving Outputs...")
    txt_filename = f"{base_name}_observation.txt"
    txt_filepath = os.path.join(OUTPUT_DIR, txt_filename)
    
    with open(txt_filepath, 'w') as f:
        f.write("--- CLINICAL AI MAMMOGRAM SCREENING REPORT ---\n")
        f.write(f"Patient ID: {patient_id}\n")
        f.write(f"View: {view_pos}\n")
        f.write(f"Source File: {filename}\n")
        f.write("----------------------------------------------\n")
        f.write(f"Cancer Probability : {cancer_prob:.1f}% ({cancer_class})\n")
        f.write(f"Breast Density     : {density_class}\n")
        f.write("----------------------------------------------\n")
        f.write("Note: Optimized via OpenCV deterministic ROI cropping.\n")
        f.write("Analyzed via ianpan/mammoscreen ensemble architecture.\n")
        
    print(f"      -> Saved report: {txt_filename}\n")
    print("Processing Complete!")

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_single_image()
import os
import glob
import pydicom
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

def crop_breast_roi(img):
    margin = 5
    if img.shape[0] <= margin * 2 or img.shape[1] <= margin * 2:
        return img
    
    img_inner = img[margin:-margin, margin:-margin]
    # Dynamic thresholding: > 5% of max pixel intensity to separate pure black
    threshold = np.max(img_inner) * 0.05
    binary_mask = (img_inner > threshold).astype("uint8")
    
    output = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
    stats = output[2]
    
    if len(stats) <= 1:
        return img
        
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    
    x1 = max(0, x1 + margin - 5)
    y1 = max(0, y1 + margin - 5)
    x2 = min(img.shape[1], x1 + w + 10)
    y2 = min(img.shape[0], y1 + h + 10)
    
    return img[y1:y2, x1:x2]

def load_mammo_model(device):
    print("\n[INIT] Initializing Ian Pan's Clinical Architecture (mammoscreen)...")
    try:
        # NOTE: You can add `local_files_only=True` here once deployed to the van for strict offline use
        model = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed to load model from Hugging Face: {e}")
        exit(1)

def get_dicom_metadata(file_path):
    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
    patient_id = getattr(ds, 'PatientID', 'Unknown_ID')
    
    view_pos = getattr(ds, 'ViewPosition', '').upper()
    if not view_pos:
        desc = getattr(ds, 'SeriesDescription', '').upper()
        view_pos = 'MLO' if 'MLO' in desc else 'CC'
        
    laterality = getattr(ds, 'ImageLaterality', getattr(ds, 'Laterality', ''))
    if not laterality:
        desc = getattr(ds, 'SeriesDescription', '').upper()
        laterality = 'R' if 'RIGHT' in desc or ' R ' in desc or desc.startswith('R') else 'L'
            
    return patient_id, laterality.upper(), view_pos

def process_batch():
    patient_folders = [f for f in glob.glob(os.path.join(INPUT_DIR, '*')) if os.path.isdir(f)]
    
    if not patient_folders:
        print(f"\n[ERROR] No patient subdirectories found in: {INPUT_DIR}")
        return

    print(f"[FOUND] {len(patient_folders)} patient study folder(s) to process.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mammo_model(device)
    
    for folder_idx, patient_dir in enumerate(patient_folders, 1):
        folder_name = os.path.basename(patient_dir)
        print(f"\n==================================================")
        print(f"[{folder_idx}/{len(patient_folders)}] Processing Study Folder: {folder_name}")
        
        patient_output_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        dcm_files = glob.glob(os.path.join(patient_dir, '*.dcm'))
        if not dcm_files:
            continue
            
        left_breast = {'cc': None, 'mlo': None}
        right_breast = {'cc': None, 'mlo': None}
        patient_id = folder_name
        
        for file_path in dcm_files:
            filename = os.path.basename(file_path)
            pid, lat, view = get_dicom_metadata(file_path)
            if pid != 'Unknown_ID': patient_id = pid
                
            print(f"  -> Reading: {filename} | Side={lat}, View={view}")
            
            try:
                img_array = model.load_image_from_dicom(file_path)
                
                img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_array = crop_breast_roi(img_array)
            except Exception as e:
                print(f"     [ERROR] Processing image array: {e}")
                continue
                
            base_name = os.path.splitext(filename)[0]
            Image.fromarray(img_array).save(os.path.join(patient_output_dir, f"{base_name}_{lat}_{view}.png"))
            
            view_key = 'mlo' if 'MLO' in view else 'cc'
            if lat == 'L': left_breast[view_key] = img_array
            else: right_breast[view_key] = img_array

        # PATCH: View Duplication to satisfy AI ensemble and prevent NoneType copy errors
        for side, b_dict in [("Left", left_breast), ("Right", right_breast)]:
            if b_dict['cc'] is not None and b_dict['mlo'] is None:
                b_dict['mlo'] = b_dict['cc']
                print(f"     [NOTE] Duplicated {side} CC to MLO to satisfy ensemble")
            elif b_dict['mlo'] is not None and b_dict['cc'] is None:
                b_dict['cc'] = b_dict['mlo']
                print(f"     [NOTE] Duplicated {side} MLO to CC to satisfy ensemble")

        inference_payloads = []
        side_labels = []
        
        # After duplication, if 'cc' exists, 'mlo' also exists. 
        if left_breast['cc'] is not None:
            inference_payloads.append(left_breast)
            side_labels.append("Left Breast (L)")
        if right_breast['cc'] is not None:
            inference_payloads.append(right_breast)
            side_labels.append("Right Breast (R)")
            
        if not inference_payloads: continue
            
        with torch.inference_mode():
            output = model(inference_payloads, device=device)
            
        txt_filepath = os.path.join(patient_output_dir, f"Study_Report_{patient_id}.txt")
        density_classes = ["A (Fatty)", "B (Scattered)", "C (Heterogeneous)", "D (Dense)"]
        
        with open(txt_filepath, 'w') as f:
            f.write("=== CLINICAL AI MAMMOGRAPHY STUDY REPORT ===\n")
            f.write(f"Patient ID : {patient_id}\n")
            f.write(f"Folder     : {folder_name}\n")
            f.write("--------------------------------------------\n\n")
            
            for idx, side_name in enumerate(side_labels):
                cancer_prob = output['cancer'][idx].item() * 100
                cancer_class = "Suspicious/Malignant" if cancer_prob > 50 else "Negative/Benign"
                
                density_idx = output['density'][idx].argmax(0).item()
                density_class = density_classes[density_idx]
                
                f.write(f"--- {side_name.upper()} ---\n")
                f.write(f"Cancer Probability : {cancer_prob:.1f}% ({cancer_class})\n")
                f.write(f"Breast Density     : {density_class}\n")
                f.write("--------------------------------------------\n\n")
                
        print(f"  [SUCCESS] Output saved to: output/{folder_name}/")

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_batch()
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

# Audit log lives at the OUTPUT_DIR root, separate from any per-patient report.
# It never contains full DICOM PatientID strings by design -- see log_id_check().
AUDIT_LOG_PATH = os.path.join(BASE_DIR, 'audit_log.txt')


def crop_breast_roi(img):
    margin = 5

    # If the DICOM loaded as a 3-channel RGB matrix (this is a 2D digital
    # mammography image, not tomosynthesis, so a 3D array here always means
    # channels-last RGB/RGBA, never a frame stack), flatten to grayscale.
    if img.ndim == 3:
        img = img[:, :, 0]

    if img.shape[0] <= margin * 2 or img.shape[1] <= margin * 2:
        return img

    img_inner = img[margin:-margin, margin:-margin]

    threshold = np.max(img_inner) * 0.05
    binary_mask = (img_inner > threshold).astype("uint8")

    output = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
    stats = output[2]

    if len(stats) <= 1:
        return img

    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]

    # FIX: coordinates from stats are relative to img_inner, which is already
    # offset from img by `margin`. The old code did `x1 + margin - 5`, which
    # with margin=5 always cancelled to `x1 + 0` -- the offset back into img
    # coordinates was never actually applied. Re-apply the margin offset,
    # then add a small fixed padding around the detected component.
    pad = 5
    x1 = max(0, (x1 + margin) - pad)
    y1 = max(0, (y1 + margin) - pad)
    x2 = min(img.shape[1], x1 + w + pad * 2)
    y2 = min(img.shape[0], y1 + h + pad * 2)

    return img[y1:y2, x1:x2]


def load_mammo_model(device):
    print("\n[INIT] Initializing Ian Pan's Clinical Architecture (mammoscreen)...")
    try:
        # NOTE: add `local_files_only=True` here once deployed to the van for strict offline use
        model = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed to load model from Hugging Face: {e}")
        exit(1)


def get_dicom_metadata(file_path):
    """
    Returns (internal_pid, laterality, view_pos).
    laterality / view_pos are 'UNK' when they can't be confidently determined --
    callers should NOT silently guess, since research datasets frequently strip
    ViewPosition/ImageLaterality/SeriesDescription entirely.
    """
    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
    internal_pid = getattr(ds, 'PatientID', None)  # may be None, blank, or a real MRN -- never used as output ID

    desc = getattr(ds, 'SeriesDescription', '').upper()

    view_pos = getattr(ds, 'ViewPosition', '').upper()
    if not view_pos:
        if 'MLO' in desc:
            view_pos = 'MLO'
        elif 'CC' in desc:
            view_pos = 'CC'
        else:
            view_pos = 'UNK'

    laterality = getattr(ds, 'ImageLaterality', getattr(ds, 'Laterality', '')).upper()
    if not laterality:
        if 'RIGHT' in desc or desc.startswith('R'):
            laterality = 'R'
        elif 'LEFT' in desc or desc.startswith('L'):
            laterality = 'L'
        else:
            laterality = 'UNK'

    return internal_pid, laterality, view_pos


def log_id_check(folder_name, internal_pids_seen):
    """
    Cross-checks the folder name (the authoritative, anonymized identifier)
    against whatever internal PatientID(s) turn up in the DICOM headers of
    files placed in that folder. This is purely a tripwire for export/copy
    mistakes (e.g. a file dragged into the wrong patient's folder) -- it is
    NOT used to name or label any output, and it is written to a standalone
    audit log, never into a per-patient report.
    """
    with open(AUDIT_LOG_PATH, 'a') as log:
        if not internal_pids_seen:
            log.write(f"{folder_name} | no internal PatientID tag found in any file (fully anonymized or stripped) | OK\n")
        elif len(internal_pids_seen) == 1:
            log.write(f"{folder_name} | 1 distinct internal PatientID present across files | OK\n")
        else:
            log.write(
                f"{folder_name} | WARNING: {len(internal_pids_seen)} DIFFERENT internal PatientIDs found "
                f"across files in this folder -- possible mis-copied/mis-sorted DICOM, check manually\n"
            )
            print(f"  [WARNING] Folder '{folder_name}' contains DICOMs from multiple internal PatientIDs. "
                  f"See {os.path.basename(AUDIT_LOG_PATH)} -- verify no files were copied into the wrong patient folder.")


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

        # The anonymized folder name is the ONLY patient identifier used anywhere
        # in outputs (filenames, report text). DICOM-internal PatientID is used
        # only for the audit cross-check above, never for naming/output.
        patient_id = folder_name

        patient_output_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(patient_output_dir, exist_ok=True)

        dcm_files = glob.glob(os.path.join(patient_dir, '*.dcm'))
        if not dcm_files:
            continue

        left_breast = {'cc': None, 'mlo': None}
        right_breast = {'cc': None, 'mlo': None}
        internal_pids_seen = set()
        failed_files = []

        for file_path in dcm_files:
            filename = os.path.basename(file_path)
            internal_pid, lat, view = get_dicom_metadata(file_path)
            if internal_pid:
                internal_pids_seen.add(internal_pid)

            print(f"  -> Reading: {filename} | Side={lat}, View={view}")

            try:
                img_array = model.load_image_from_dicom(file_path)
                img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_array = crop_breast_roi(img_array)
            except Exception as e:
                print(f"     [ERROR] Processing image array: {e}")
                failed_files.append(filename)
                continue

            base_name = os.path.splitext(filename)[0]
            Image.fromarray(img_array).save(os.path.join(patient_output_dir, f"{base_name}_{lat}_{view}.png"))

            # Only bucket into CC/MLO slots when the view is confidently known.
            # 'UNK' views are still saved as PNGs above (so you can eyeball and
            # relabel them) but are NOT fed into the CC/MLO ensemble slots below,
            # since guessing wrong here would silently corrupt which slot the
            # model treats as which view.
            if view == 'UNK' or lat == 'UNK':
                print(f"     [NOTE] {filename} saved but excluded from AI ensemble input (side/view undetermined)")
                continue

            view_key = 'mlo' if view == 'MLO' else 'cc'
            if lat == 'L':
                left_breast[view_key] = img_array
            else:
                right_breast[view_key] = img_array

        log_id_check(folder_name, internal_pids_seen)

        # View duplication for single-view test data / incomplete studies.
        # Track which sides were duplicated so the report can flag reduced confidence.
        duplicated_sides = []
        for side, b_dict in [("Left", left_breast), ("Right", right_breast)]:
            if b_dict['cc'] is not None and b_dict['mlo'] is None:
                b_dict['mlo'] = b_dict['cc']
                duplicated_sides.append(side)
                print(f"     [NOTE] Duplicated {side} CC to MLO to satisfy ensemble (single view available)")
            elif b_dict['mlo'] is not None and b_dict['cc'] is None:
                b_dict['cc'] = b_dict['mlo']
                duplicated_sides.append(side)
                print(f"     [NOTE] Duplicated {side} MLO to CC to satisfy ensemble (single view available)")

        inference_payloads = []
        side_labels = []

        if left_breast['cc'] is not None:
            inference_payloads.append(left_breast)
            side_labels.append("Left Breast (L)")
        if right_breast['cc'] is not None:
            inference_payloads.append(right_breast)
            side_labels.append("Right Breast (R)")

        if not inference_payloads:
            print(f"  [SKIP] No usable CC/MLO pair for {folder_name} -- nothing to run through the ensemble.")
            continue

        with torch.inference_mode():
            output = model(inference_payloads, device=device)

        txt_filepath = os.path.join(patient_output_dir, f"Study_Report_{patient_id}.txt")
        density_classes = ["A (Fatty)", "B (Scattered)", "C (Heterogeneous)", "D (Dense)"]

        with open(txt_filepath, 'w') as f:
            f.write("=== CLINICAL AI MAMMOGRAPHY STUDY REPORT ===\n")
            f.write(f"Patient ID : {patient_id}\n")
            f.write(f"Folder     : {folder_name}\n")
            if failed_files:
                f.write(f"WARNING    : {len(failed_files)} file(s) failed to process and were excluded: "
                         f"{', '.join(failed_files)}\n")
            f.write("--------------------------------------------\n\n")

            for idx, side_name in enumerate(side_labels):
                cancer_prob = output['cancer'][idx].item() * 100
                cancer_class = "Suspicious/Malignant" if cancer_prob > 50 else "Negative/Benign"

                density_idx = output['density'][idx].argmax(0).item()
                density_class = density_classes[density_idx]

                f.write(f"--- {side_name.upper()} ---\n")
                f.write(f"Cancer Probability : {cancer_prob:.1f}% ({cancer_class})\n")
                f.write(f"Breast Density     : {density_class}\n")

                side_word = side_name.split()[0]  # "Left" or "Right"
                if side_word in duplicated_sides:
                    f.write("NOTE: Only a single view was available for this side. The second view\n")
                    f.write("was duplicated to satisfy the model's two-view input requirement.\n")
                    f.write("Treat this result as lower-confidence than a true two-view study.\n")

                f.write("--------------------------------------------\n\n")

        print(f"  [SUCCESS] Output saved to: output/{folder_name}/")


if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_batch()
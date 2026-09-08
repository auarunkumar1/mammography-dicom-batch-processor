"""
Batch DICOM Mammography Processor (v3)
=======================================

A local-first clinical AI screening pipeline for digital mammography. It
processes whole patient study folders of DICOM images and produces, per
breast, a malignancy probability and an ACR breast density class (A-D),
saved as a per-patient text report plus per-view PNG previews.

CREDIT / ACKNOWLEDGMENT
------------------------
The AI engine behind this pipeline is Dr. Ian Pan's open-source
"mammoscreen" model, publicly available on the Hugging Face Hub:

    * Personal website : https://ianpan.me
    * GitHub           : https://i-pan.github.io
    * Hugging Face     : https://huggingface.co/ianpan
    * Model card       : https://huggingface.co/ianpan/mammoscreen

The model was trained and validated on the global RSNA Screening
Mammography Challenge dataset (54,000+ images) and simultaneously outputs
malignancy probability and ACR breast density classification. It is used
here under its original open-access terms; see the model card above for the
exact license and usage terms.
"""

import os
import glob
import re
from datetime import datetime
from collections import Counter
import pydicom
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Audit log lives at the OUTPUT_DIR root, separate from any per-patient report.
# It never contains full DICOM PatientID strings by design -- see log_id_check().
AUDIT_LOG_PATH = os.path.join(BASE_DIR, 'audit_log.txt')
SUMMARY_EXCEL_PATH = os.path.join(OUTPUT_DIR, 'mammography_summary.xlsx')


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


def format_patient_age(age_val, birth_date=None, study_date=None):
    if age_val:
        s = str(age_val).strip()
        m = re.match(r"^0*(\d+)\s*([YyMmDdWw]?)$", s)
        if m:
            num, unit = m.groups()
            unit = unit.upper() if unit else "Y"
            if num == "0":
                return "N/A"
            return num if unit == "Y" else f"{num}{unit}"
        return s if s != "0" else "N/A"
    if birth_date and study_date:
        try:
            b_dt = datetime.strptime(str(birth_date).strip(), "%Y%m%d")
            s_dt = datetime.strptime(str(study_date).strip(), "%Y%m%d")
            age = s_dt.year - b_dt.year - ((s_dt.month, s_dt.day) < (b_dt.month, b_dt.day))
            return str(age) if age > 0 else "N/A"
        except Exception:
            pass
    return "N/A"


def format_patient_name(name_val):
    if not name_val:
        return ""
    s = str(name_val).replace('^', ' ').strip()
    s = " ".join(s.split())
    if not s or s.upper() in ("ANONYMOUS", "ANON", "UNKNOWN", "NONE"):
        return ""
    return s


def format_dicom_date(date_val):
    if not date_val:
        return ""
    s = str(date_val).strip()
    if len(s) == 8 and s.isdigit():
        try:
            dt = datetime.strptime(s, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return s
    return s


def extract_study_demographics(dcm_files):
    name = ""
    age = "N/A"
    scan_date = "Unknown Date"

    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if not name:
                p_name = ds.get('PatientName', None)
                if p_name:
                    name = format_patient_name(p_name)
            if age == "N/A":
                p_age = ds.get('PatientAge', None)
                b_date = ds.get('PatientBirthDate', None)
                s_date = ds.get('StudyDate', None)
                cleaned_age = format_patient_age(p_age, b_date, s_date)
                if cleaned_age and cleaned_age != "N/A":
                    age = cleaned_age
            if scan_date == "Unknown Date":
                for date_tag in ['StudyDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate']:
                    d_val = ds.get(date_tag, None)
                    if d_val:
                        cleaned_date = format_dicom_date(d_val)
                        if cleaned_date:
                            scan_date = cleaned_date
                            break
        except Exception:
            continue
    return name, age, scan_date


def export_to_excel(study_records, excel_path):
    if not study_records:
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Mammography Summary"
    ws.views.sheetView[0].showGridLines = True

    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_header = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")

    font_date_group = Font(name="Calibri", size=11, bold=True, color="1F4E79")
    fill_date_group = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    font_data = Font(name="Calibri", size=11)
    font_alert = Font(name="Calibri", size=11, bold=True, color="9C0006")
    fill_alert = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    font_warning = Font(name="Calibri", size=11, color="9C6500")
    fill_warning = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    border_thin = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9")
    )
    border_date_group = Border(
        top=Side(style="medium", color="8EA9DB"),
        bottom=Side(style="medium", color="8EA9DB")
    )

    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")

    headers = [
        "Scanned Date",
        "Patient ID",
        "Patient Name",
        "Age",
        "Left Malignancy (%)",
        "Left Breast Density",
        "Left Finding",
        "Right Malignancy (%)",
        "Right Breast Density",
        "Right Finding",
        "Single View Duplication",
        "Flagged (>5%)",
        "Status / Notes"
    ]

    ws.append(headers)
    ws.row_dimensions[1].height = 26
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_center

    def sort_key(rec):
        d = rec.get("scanned_date", "")
        date_sort = ("9999-99-99" if not d or "Unknown" in d else d)
        return (date_sort, rec.get("patient_id", ""))

    def apply_data_row(target_ws, r_idx, row_data, l_prob, r_prob, is_flagged):
        for c_idx, val in enumerate(row_data, 1):
            c = target_ws.cell(row=r_idx, column=c_idx, value=val)
            c.font = font_data
            c.border = border_thin

            if c_idx in (1, 2, 4, 6, 9, 11, 12):
                c.alignment = align_center
            elif c_idx in (5, 8):
                c.alignment = align_right
            else:
                c.alignment = align_left

            # Highlight elevated cancer probability (>= 5.0%)
            if c_idx == 5 and l_prob is not None and l_prob >= 5.0:
                c.fill = fill_alert
                c.font = font_alert
            elif c_idx == 8 and r_prob is not None and r_prob >= 5.0:
                c.fill = fill_alert
                c.font = font_alert

            # Highlight Single View duplication
            if c_idx == 11 and val != "None":
                c.fill = fill_warning
                c.font = font_warning

            # Highlight Flagged column
            if c_idx == 12 and is_flagged:
                c.fill = fill_alert
                c.font = font_alert
        target_ws.row_dimensions[r_idx].height = 20

    def autofit_cols(target_ws):
        for col in target_ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.coordinate in target_ws.merged_cells:
                    continue
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            target_ws.column_dimensions[col_letter].width = max(max_len + 3, 13)

    sorted_records = sorted(study_records, key=sort_key)
    date_counts = Counter(r.get("scanned_date", "Unknown Date") for r in sorted_records)

    current_date = None
    current_row = 2

    for rec in sorted_records:
        rec_date = rec.get("scanned_date", "Unknown Date")

        # Insert date group banner when date changes
        if rec_date != current_date:
            current_date = rec_date
            count = date_counts[current_date]
            banner_text = f"Scanned Date: {current_date}  ({count} stud{'ies' if count > 1 else 'y'})"

            ws.cell(row=current_row, column=1, value=banner_text)
            ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=len(headers))

            for c in range(1, len(headers) + 1):
                b_cell = ws.cell(row=current_row, column=c)
                b_cell.font = font_date_group
                b_cell.fill = fill_date_group
                b_cell.border = border_date_group
                b_cell.alignment = align_left
            ws.row_dimensions[current_row].height = 22
            current_row += 1

        left_prob = rec.get("left_prob")
        right_prob = rec.get("right_prob")
        left_prob_val = f"{left_prob:.1f}%" if left_prob is not None else "N/A"
        right_prob_val = f"{right_prob:.1f}%" if right_prob is not None else "N/A"
        flagged = rec.get("flagged", False)

        row_vals = [
            rec.get("scanned_date") or "Unknown Date",
            rec.get("patient_id") or "",
            rec.get("patient_name") or "N/A",
            rec.get("patient_age") or "N/A",
            left_prob_val,
            rec.get("left_density") or "N/A",
            rec.get("left_finding") or "N/A",
            right_prob_val,
            rec.get("right_density") or "N/A",
            rec.get("right_finding") or "N/A",
            rec.get("single_view") or "None",
            "YES" if flagged else "NO",
            rec.get("notes") or "OK"
        ]

        apply_data_row(ws, current_row, row_vals, left_prob, right_prob, flagged)
        current_row += 1

    ws.freeze_panes = "A2"
    autofit_cols(ws)

    # If multiple distinct dates exist, also create individual tabs per date
    if len(date_counts) > 1:
        for d_key in sorted(date_counts.keys(), key=lambda x: ("9999-99-99" if "Unknown" in x or not x else x)):
            d_records = [r for r in sorted_records if r.get("scanned_date", "Unknown Date") == d_key]
            tab_title = re.sub(r'[\\/*?:\[\]]', '_', d_key)[:31]
            ws_d = wb.create_sheet(title=tab_title)
            ws_d.views.sheetView[0].showGridLines = True

            ws_d.append(headers)
            ws_d.row_dimensions[1].height = 26
            for col_idx in range(1, len(headers) + 1):
                cell = ws_d.cell(row=1, column=col_idx)
                cell.font = font_header
                cell.fill = fill_header
                cell.alignment = align_center

            for d_row_idx, d_rec in enumerate(d_records, 2):
                l_prob = d_rec.get("left_prob")
                r_prob = d_rec.get("right_prob")
                l_prob_val = f"{l_prob:.1f}%" if l_prob is not None else "N/A"
                r_prob_val = f"{r_prob:.1f}%" if r_prob is not None else "N/A"
                d_flagged = d_rec.get("flagged", False)

                d_row_vals = [
                    d_rec.get("scanned_date") or "Unknown Date",
                    d_rec.get("patient_id") or "",
                    d_rec.get("patient_name") or "N/A",
                    d_rec.get("patient_age") or "N/A",
                    l_prob_val,
                    d_rec.get("left_density") or "N/A",
                    d_rec.get("left_finding") or "N/A",
                    r_prob_val,
                    d_rec.get("right_density") or "N/A",
                    d_rec.get("right_finding") or "N/A",
                    d_rec.get("single_view") or "None",
                    "YES" if d_flagged else "NO",
                    d_rec.get("notes") or "OK"
                ]
                apply_data_row(ws_d, d_row_idx, d_row_vals, l_prob, r_prob, d_flagged)

            ws_d.freeze_panes = "A2"
            autofit_cols(ws_d)

    try:
        wb.save(excel_path)
        print(f"\n[EXCEL] Consolidated summary successfully exported to: {excel_path}")
    except PermissionError:
        fallback_path = os.path.join(
            OUTPUT_DIR, f"mammography_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        wb.save(fallback_path)
        print(f"\n[WARNING] Could not overwrite {excel_path} (file open elsewhere). Saved to: {fallback_path}")


def process_batch():
    patient_folders = [f for f in glob.glob(os.path.join(INPUT_DIR, '*')) if os.path.isdir(f)]

    if not patient_folders:
        print(f"\n[ERROR] No patient subdirectories found in: {INPUT_DIR}")
        return

    print(f"[FOUND] {len(patient_folders)} patient study folder(s) to process.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mammo_model(device)
    study_records = []

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

        patient_name, patient_age, scanned_date = extract_study_demographics(dcm_files)
        print(f"  -> Demographics: Name='{patient_name or 'N/A'}' | Age={patient_age} | Date={scanned_date}")

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

        left_prob = None
        left_density = None
        left_class = None
        right_prob = None
        right_density = None
        right_class = None

        with open(txt_filepath, 'w') as f:
            f.write("=== CLINICAL AI MAMMOGRAPHY STUDY REPORT ===\n")
            f.write(f"Patient ID   : {patient_id}\n")
            f.write(f"Folder       : {folder_name}\n")
            if patient_name:
                f.write(f"Patient Name : {patient_name}\n")
            if patient_age != "N/A":
                f.write(f"Age          : {patient_age}\n")
            if scanned_date != "Unknown Date":
                f.write(f"Scanned Date : {scanned_date}\n")
            if failed_files:
                f.write(f"WARNING      : {len(failed_files)} file(s) failed to process and were excluded: "
                         f"{', '.join(failed_files)}\n")
            f.write("--------------------------------------------\n\n")

            for idx, side_name in enumerate(side_labels):
                cancer_prob = output['cancer'][idx].item() * 100
                cancer_class = "Suspicious/Malignant" if cancer_prob > 50 else "Negative/Benign"

                density_idx = output['density'][idx].argmax(0).item()
                density_class = density_classes[density_idx]

                if "Left" in side_name:
                    left_prob = cancer_prob
                    left_density = density_class
                    left_class = cancer_class
                elif "Right" in side_name:
                    right_prob = cancer_prob
                    right_density = density_class
                    right_class = cancer_class

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

        study_records.append({
            "scanned_date": scanned_date,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "left_prob": left_prob,
            "left_density": left_density,
            "left_finding": left_class,
            "right_prob": right_prob,
            "right_density": right_density,
            "right_finding": right_class,
            "single_view": ", ".join(duplicated_sides) if duplicated_sides else "None",
            "flagged": (left_prob is not None and left_prob >= 5.0) or (right_prob is not None and right_prob >= 5.0),
            "notes": f"{len(failed_files)} file(s) failed" if failed_files else "OK"
        })

    if study_records:
        export_to_excel(study_records, SUMMARY_EXCEL_PATH)


if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_batch()
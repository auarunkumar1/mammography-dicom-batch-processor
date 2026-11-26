#!/usr/bin/env python3
"""
Batch DICOM Mammography Processor (v4)
=======================================

A local-first clinical AI screening and reporting pipeline for digital mammography.

WHAT'S NEW IN V4:
-----------------
1. Automated DICOM Segregation from Single-Folder / Date-Range Exports:
   - Ingests a single flat folder (or mixed directory) of DICOM files exported from
     PACS/workstation across a date range.
   - Automatically inspects DICOM headers (StudyInstanceUID, PatientID, PatientName,
     StudyDate, StudyTime) and groups files into distinct patient studies in memory.
   - Automatically derives clean, standardized patient folder names (e.g. abi23-umamageswari).
   - Fully backward-compatible with pre-organized patient subfolders (v3 behavior).
2. Date Range Filtering:
   - Optional CLI flags (--start-date and --end-date) to filter examinations.
3. Optional Input Physical Organization:
   - Optional CLI flag (--organize-input) to copy/move raw DICOMs into segregated subfolders.
4. Integrated High-Resolution 4-View Clinical Reporting:
   - Automatically generates professional vector PDF reports for all flagged patients (>5% risk).
   - Embeds native high-resolution 4-view mammogram projections (RCC, LCC, RMLO, LMLO).
   - Multi-page layout: Page 1 (Overview + 4 views), Page 2 (CC pair), Page 3 (MLO pair).
   - STRICTLY NO AI model or architecture details mentioned in reports.
   - Generates consolidated multi-patient PDF and Excel summary automatically.

CREDIT / ACKNOWLEDGMENT
------------------------
The AI engine behind this pipeline is Dr. Ian Pan's open-source "mammoscreen" model,
publicly available on the Hugging Face Hub (https://huggingface.co/ianpan/mammoscreen).
"""

import os
import sys
import glob
import re
import argparse
import subprocess
import shutil
from datetime import datetime
from collections import Counter, defaultdict
import pydicom
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ReportLab & PDF imports
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from pypdf import PdfWriter
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, 'input')
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
AUDIT_LOG_PATH = os.path.join(BASE_DIR, 'audit_log.txt')


# =====================================================================
# 1. IMAGE PROCESSING & ROI CROPPING
# =====================================================================

def crop_breast_roi(img):
    """
    Isolates the breast tissue using connected component analysis,
    removing black background padding, dead space, and burned-in text.
    """
    margin = 5
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

    pad = 5
    x1 = max(0, (x1 + margin) - pad)
    y1 = max(0, (y1 + margin) - pad)
    x2 = min(img.shape[1], x1 + w + pad * 2)
    y2 = min(img.shape[0], y1 + h + pad * 2)

    return img[y1:y2, x1:x2]


# =====================================================================
# 2. AI MODEL LOADING
# =====================================================================

def load_mammo_model(device):
    print("\n[INIT] Initializing Clinical Architecture (mammoscreen)...")
    try:
        model = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed to load model from Hugging Face: {e}")
        sys.exit(1)


# =====================================================================
# 3. DICOM METADATA & DEMOGRAPHIC PARSING
# =====================================================================

def format_patient_name(name_val):
    if not name_val:
        return ""
    s = str(name_val).replace('^', ' ').strip()
    s = " ".join(s.split())
    if not s or s.upper() in ("ANONYMOUS", "ANON", "UNKNOWN", "NONE"):
        return ""
    return s


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


def format_dicom_time(time_val):
    if not time_val:
        return ""
    s = str(time_val).split('.')[0].strip().zfill(6)
    if len(s) == 6 and s.isdigit():
        return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"
    return s


def get_dicom_view_metadata(file_path):
    """Returns (internal_pid, laterality, view_pos)."""
    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
    internal_pid = getattr(ds, 'PatientID', None)
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


def make_patient_slug(patient_id, patient_name, study_date=None):
    """
    Generates a standardized clinical study folder slug:
    e.g. 'abi23-umamageswari' or 'vlr35-geetha-d'.
    """
    clean_id = re.sub(r'[^a-zA-Z0-9]+', '-', str(patient_id or '').strip()).strip('-').lower()
    pname_raw = str(patient_name or '').strip()
    pname_clean = re.sub(r'^(MRS|MR|DR|MS|MISS|MASTER)[\s\.\^]+', '', pname_raw, flags=re.IGNORECASE)
    pname_slug = re.sub(r'[^a-zA-Z0-9]+', '-', pname_clean).strip('-').lower()

    if clean_id and pname_slug:
        return f"{clean_id}-{pname_slug}"
    elif clean_id:
        return clean_id
    elif pname_slug:
        return f"patient-{pname_slug}"
    elif study_date:
        return f"study-{study_date}"
    return "unknown-study"


# =====================================================================
# 4. DISCOVERY & AUTO-SEGREGATION ENGINE
# =====================================================================

def discover_dicom_studies(input_dir, start_date=None, end_date=None, organize_input=False):
    """
    Intelligently inspects input_dir. Supports:
      A. Pre-organized subfolders (legacy mode: input/<patient_id>/*.dcm)
      B. Single-folder flat export (all DICOMs in input_dir/*.dcm)
      C. Nested / mixed directory structures
    Groups files by StudyInstanceUID and returns a list of study bundles.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Check for immediate subdirectories containing DICOM files
    subdirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')]
    
    # Check for flat DICOM files directly in input_dir
    flat_dcms = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.dcm', '.ima'))]

    # Mode A: If multiple subdirectories exist and each contains .dcm, treat as pre-organized
    if subdirs and not flat_dcms:
        print(f"[DISCOVERY] Found {len(subdirs)} pre-organized patient subdirectories in: {input_dir}")
        studies = []
        for sdir in sorted(subdirs):
            dcms = glob.glob(os.path.join(sdir, '*.dcm')) + glob.glob(os.path.join(sdir, '*.ima'))
            if not dcms:
                continue
            folder_name = os.path.basename(sdir)
            p_name, p_age, p_date, p_time, p_id, study_uid = inspect_dicom_bundle(dcms, fallback_folder=folder_name)
            
            # Apply date filter
            if not date_matches(p_date, start_date, end_date):
                continue

            studies.append({
                'study_uid': study_uid or folder_name,
                'folder_name': folder_name,
                'patient_id': p_id or folder_name,
                'patient_name': p_name,
                'patient_age': p_age,
                'scanned_date': p_date,
                'scanned_time': p_time,
                'dcm_files': dcms
            })
        return studies

    # Mode B: Single flat folder or mixed tree -- Auto-Segregate by DICOM metadata
    print(f"[DISCOVERY] Auto-segregating DICOM files based on embedded metadata in: {input_dir}")
    all_dcms = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.dcm', '.ima')) or (len(f.split('.')) > 4 and 'dcm' in f.lower()):
                all_dcms.append(os.path.join(root, f))

    if not all_dcms:
        # Fallback: check without extension if DICM magic bytes present
        for root, _, files in os.walk(input_dir):
            for f in files:
                fpath = os.path.join(root, f)
                if os.path.isfile(fpath) and os.path.getsize(fpath) > 132:
                    try:
                        with open(fpath, 'rb') as fp:
                            fp.seek(128)
                            if fp.read(4) == b'DICM':
                                all_dcms.append(fpath)
                    except Exception:
                        pass

    print(f"  -> Discovered {len(all_dcms)} total DICOM file(s). Inspecting headers...")

    # Group by StudyInstanceUID
    grouped_by_uid = defaultdict(list)
    for fpath in all_dcms:
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            suid = getattr(ds, 'StudyInstanceUID', None)
            if not suid:
                # Fallback to (PatientID, StudyDate)
                pid = getattr(ds, 'PatientID', 'UNKNOWN_PID')
                sdate = getattr(ds, 'StudyDate', 'UNKNOWN_DATE')
                suid = f"{pid}_{sdate}"
            grouped_by_uid[str(suid)].append((fpath, ds))
        except Exception as e:
            print(f"  [Warning] Could not parse DICOM header for {os.path.basename(fpath)}: {e}")

    print(f"  -> Successfully segregated into {len(grouped_by_uid)} distinct clinical study group(s).")

    studies = []
    used_folders = set()

    for suid, items in sorted(grouped_by_uid.items()):
        dcms = [it[0] for it in items]
        ds_sample = items[0][1]

        p_name = format_patient_name(getattr(ds_sample, 'PatientName', ''))
        p_id = str(getattr(ds_sample, 'PatientID', '')).strip()
        p_age = format_patient_age(
            getattr(ds_sample, 'PatientAge', None),
            getattr(ds_sample, 'PatientBirthDate', None),
            getattr(ds_sample, 'StudyDate', None)
        )
        s_date_raw = getattr(ds_sample, 'StudyDate', None) or getattr(ds_sample, 'SeriesDate', None)
        p_date = format_dicom_date(s_date_raw) or "Unknown Date"

        s_time_raw = getattr(ds_sample, 'StudyTime', None) or getattr(ds_sample, 'SeriesTime', None) or getattr(ds_sample, 'AcquisitionTime', None)
        p_time = format_dicom_time(s_time_raw) or "Unknown Time"

        # Apply date filter
        if not date_matches(p_date, start_date, end_date):
            continue

        folder_name = make_patient_slug(p_id, p_name, p_date)
        # Collision avoidance
        base_f = folder_name
        ctr = 2
        while folder_name in used_folders:
            folder_name = f"{base_f}-{ctr}"
            ctr += 1
        used_folders.add(folder_name)

        # Optional: physically organize inputs into subfolders
        final_dcms = dcms
        if organize_input:
            target_subfolder = os.path.join(input_dir, folder_name)
            os.makedirs(target_subfolder, exist_ok=True)
            moved_dcms = []
            for src_dcm in dcms:
                dst_dcm = os.path.join(target_subfolder, os.path.basename(src_dcm))
                if src_dcm != dst_dcm:
                    shutil.copyfile(src_dcm, dst_dcm)
                moved_dcms.append(dst_dcm)
            final_dcms = moved_dcms

        studies.append({
            'study_uid': suid,
            'folder_name': folder_name,
            'patient_id': p_id or folder_name,
            'patient_name': p_name,
            'patient_age': p_age,
            'scanned_date': p_date,
            'scanned_time': p_time,
            'dcm_files': final_dcms
        })

    return studies


def inspect_dicom_bundle(dcm_files, fallback_folder=''):
    name, age, scan_date, scan_time, pid, study_uid = "", "N/A", "Unknown Date", "Unknown Time", "", None
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if not study_uid:
                study_uid = getattr(ds, 'StudyInstanceUID', None)
            if not pid:
                p_id_raw = getattr(ds, 'PatientID', None)
                if p_id_raw:
                    pid = str(p_id_raw).strip()
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
            if scan_time == "Unknown Time":
                for time_tag in ['StudyTime', 'SeriesTime', 'AcquisitionTime', 'ContentTime']:
                    t_val = ds.get(time_tag, None)
                    if t_val:
                        cleaned_time = format_dicom_time(t_val)
                        if cleaned_time:
                            scan_time = cleaned_time
                            break
        except Exception:
            continue
    return name, age, scan_date, scan_time, pid or fallback_folder, study_uid


def date_matches(scanned_date, start_date=None, end_date=None):
    if not start_date and not end_date:
        return True
    if not scanned_date or scanned_date == "Unknown Date":
        return False
    try:
        cur_dt = datetime.strptime(scanned_date, "%Y-%m-%d").date()
        if start_date:
            st_dt = datetime.strptime(start_date.replace('/', '-'), "%Y-%m-%d").date() if '-' in start_date or '/' in start_date else datetime.strptime(start_date, "%Y%m%d").date()
            if cur_dt < st_dt:
                return False
        if end_date:
            en_dt = datetime.strptime(end_date.replace('/', '-'), "%Y-%m-%d").date() if '-' in end_date or '/' in end_date else datetime.strptime(end_date, "%Y%m%d").date()
            if cur_dt > en_dt:
                return False
        return True
    except Exception:
        return True


# =====================================================================
# 5. AUDIT LOGGING
# =====================================================================

def log_id_check(folder_name, internal_pids_seen):
    with open(AUDIT_LOG_PATH, 'a') as log:
        if not internal_pids_seen:
            log.write(f"{folder_name} | no internal PatientID tag found in any file (fully anonymized) | OK\n")
        elif len(internal_pids_seen) == 1:
            log.write(f"{folder_name} | 1 distinct internal PatientID present across files | OK\n")
        else:
            log.write(
                f"{folder_name} | WARNING: {len(internal_pids_seen)} DIFFERENT internal PatientIDs found "
                f"across files in this study bundle -- verify sorting\n"
            )


# =====================================================================
# 6. EXCEL SUMMARY EXPORT
# =====================================================================

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
        "Scanned Date", "Patient ID", "Patient Name", "Age",
        "Left Malignancy (%)", "Left Breast Density", "Left Finding",
        "Right Malignancy (%)", "Right Breast Density", "Right Finding",
        "Single View Duplication", "Flagged (>5%)", "Status / Notes"
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

            if c_idx == 5 and l_prob is not None and l_prob >= 5.0:
                c.fill = fill_alert
                c.font = font_alert
            elif c_idx == 8 and r_prob is not None and r_prob >= 5.0:
                c.fill = fill_alert
                c.font = font_alert

            if c_idx == 11 and val != "None":
                c.fill = fill_warning
                c.font = font_warning

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

    # Individual tabs per date if multiple dates exist
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
            os.path.dirname(excel_path), f"mammography_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        wb.save(fallback_path)
        print(f"\n[WARNING] Could not overwrite {excel_path}. Saved to: {fallback_path}")


def generate_summary_pdf(output_dir):
    summary_excel_path = os.path.join(output_dir, 'mammography_summary.xlsx')
    if not os.path.exists(summary_excel_path):
        return

    print("\n[PDF] Generating mammography_summary.pdf (Landscape, Column B removed)...")
    wb = openpyxl.load_workbook(summary_excel_path)
    for sheetname in wb.sheetnames:
        if sheetname != 'Mammography Summary':
            wb.remove(wb[sheetname])

    ws = wb['Mammography Summary']
    banner_rows = [rng.min_row for rng in ws.merged_cells.ranges if rng.min_row == rng.max_row]
    for rng in list(ws.merged_cells.ranges):
        ws.unmerge_cells(str(rng))

    ws.delete_cols(2)  # Delete Column B
    for r in banner_rows:
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=12)

    ws.column_dimensions.clear()
    col_widths = {
        'A': 15, 'B': 28, 'C': 10, 'D': 20, 'E': 22, 'F': 18,
        'G': 20, 'H': 22, 'I': 18, 'J': 24, 'K': 16, 'L': 16
    }
    for col_letter, w in col_widths.items():
        ws.column_dimensions[col_letter].width = w

    ws.print_title_rows = '1:1'
    ws.page_setup.orientation = ws.ORIENTATION_LANDSCAPE
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True

    temp_xlsx = os.path.join('/tmp', 'mammography_summary_master_only.xlsx')
    wb.save(temp_xlsx)

    try:
        subprocess.run([
            'libreoffice', '--headless', '--convert-to', 'pdf',
            '--outdir', output_dir, temp_xlsx
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        target_pdf = os.path.join(output_dir, 'mammography_summary.pdf')
        gen_pdf = os.path.join(output_dir, 'mammography_summary_master_only.xlsx.pdf')
        alt_gen_pdf = os.path.join(output_dir, 'mammography_summary_master_only.pdf')
        if os.path.exists(gen_pdf):
            shutil.move(gen_pdf, target_pdf)
        elif os.path.exists(alt_gen_pdf):
            shutil.move(alt_gen_pdf, target_pdf)
        print(f"  [OK] Master summary PDF created: {target_pdf}")
    except Exception as e:
        print(f"  [Skip] LibreOffice headless export unavailable: {e}")


# =====================================================================
# 7. HIGH-RESOLUTION CLINICAL PDF REPORT GENERATION (NO AI MENTION)
# =====================================================================

def render_high_res_patient_pdf(pdf_path, patient_rec, pdir=None):
    if not HAS_REPORTLAB:
        print("  [Skip] reportlab or pypdf not installed; skipping PDF generation.")
        return

    c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
    W, H = landscape(A4)

    pid = patient_rec['patient_id']
    name = patient_rec['patient_name']
    s_date = patient_rec['scanned_date']
    s_time = patient_rec.get('scanned_time', 'N/A')
    age = patient_rec.get('patient_age', 'N/A')

    l_p_val = patient_rec.get('left_prob') or 0.0
    r_p_val = patient_rec.get('right_prob') or 0.0
    l_flag = l_p_val >= 5.0
    r_flag = r_p_val >= 5.0
    is_study_flagged = l_flag or r_flag

    l_prob_str = f"{l_p_val:.1f}%" if patient_rec.get('left_prob') is not None else "N/A"
    r_prob_str = f"{r_p_val:.1f}%" if patient_rec.get('right_prob') is not None else "N/A"
    l_dens = patient_rec.get('left_density', 'N/A')
    r_dens = patient_rec.get('right_density', 'N/A')
    l_find = patient_rec.get('left_finding', 'Negative/Benign')
    r_find = patient_rec.get('right_finding', 'Negative/Benign')

    imgs = patient_rec.get('image_paths', {})
    if not imgs and pdir and os.path.exists(pdir):
        pngs = glob.glob(os.path.join(pdir, '*.png'))
        def find_img(suf):
            m = [p for p in pngs if suf in os.path.basename(p) and '_Ann_' not in os.path.basename(p)]
            if not m:
                m = [p for p in pngs if suf in os.path.basename(p)]
            return m[0] if m else None
        imgs = {
            'R_CC': find_img('_R_CC.png'),
            'L_CC': find_img('_L_CC.png'),
            'R_MLO': find_img('_R_MLO.png'),
            'L_MLO': find_img('_L_MLO.png'),
        }

    # ---------------- PAGE 1: 4-VIEW CLINICAL OVERVIEW ----------------
    c.setFillColor(colors.HexColor('#0F2942'))
    c.rect(0, H - 50, W, 50, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 16)
    c.drawString(25, H - 33, 'DIGITAL MAMMOGRAPHY EXAMINATION REPORT')

    badge_w, badge_h = 280, 28
    badge_x, badge_y = W - 25 - badge_w, H - 39
    if is_study_flagged:
        c.setFillColor(colors.HexColor('#FEE2E2'))
        c.setStrokeColor(colors.HexColor('#DC2626'))
        c.setLineWidth(1.5)
        c.roundRect(badge_x, badge_y, badge_w, badge_h, 4, fill=1, stroke=1)
        c.setFillColor(colors.HexColor('#991B1B'))
        c.setFont('Helvetica-Bold', 10.5)
        c.drawCentredString(badge_x + badge_w / 2.0, badge_y + 8.5, 'FLAGGED STUDY : MALIGNANCY RISK > 5%')
    else:
        c.setFillColor(colors.HexColor('#DCFCE7'))
        c.setStrokeColor(colors.HexColor('#16A34A'))
        c.setLineWidth(1.5)
        c.roundRect(badge_x, badge_y, badge_w, badge_h, 4, fill=1, stroke=1)
        c.setFillColor(colors.HexColor('#166534'))
        c.setFont('Helvetica-Bold', 10.5)
        c.drawCentredString(badge_x + badge_w / 2.0, badge_y + 8.5, 'ROUTINE STUDY : WITHIN NORMAL LIMITS')

    # Demographics
    demo_y, demo_h = H - 116, 60
    c.setFillColor(colors.HexColor('#F8FAFC'))
    c.setStrokeColor(colors.HexColor('#CBD5E1'))
    c.setLineWidth(1)
    c.roundRect(25, demo_y, W - 50, demo_h, 5, fill=1, stroke=1)

    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont('Helvetica-Bold', 8)
    c.drawString(38, demo_y + 44, 'PATIENT NAME')
    c.drawString(38, demo_y + 19, 'PATIENT ID / FOLDER')

    c.setFillColor(colors.HexColor('#0F172A'))
    c.setFont('Helvetica-Bold', 11)
    c.drawString(38, demo_y + 31, str(name or 'N/A').upper())
    c.setFont('Helvetica', 10)
    c.drawString(38, demo_y + 7, str(pid))

    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont('Helvetica-Bold', 8)
    c.drawString(275, demo_y + 44, 'STUDY DATE')
    c.drawString(275, demo_y + 19, 'STUDY TIME')

    c.setFillColor(colors.HexColor('#0F172A'))
    c.setFont('Helvetica-Bold', 10.5)
    c.drawString(275, demo_y + 31, str(s_date))
    c.drawString(275, demo_y + 7, str(s_time))

    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont('Helvetica-Bold', 8)
    c.drawString(460, demo_y + 44, 'AGE / GENDER')
    c.drawString(460, demo_y + 19, 'EXAMINATION')

    c.setFillColor(colors.HexColor('#0F172A'))
    c.setFont('Helvetica-Bold', 10.5)
    age_str = f"{age} Y" if str(age).isdigit() else str(age)
    c.drawString(460, demo_y + 31, f"{age_str}  /  Female")
    c.setFont('Helvetica', 9.5)
    c.drawString(460, demo_y + 7, 'Digital Bilateral Mammogram (4 Views)')

    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont('Helvetica-Bold', 8)
    c.drawString(645, demo_y + 44, 'IMAGING MODALITY')
    c.drawString(645, demo_y + 19, 'CLINICAL CLASSIFICATION')

    c.setFillColor(colors.HexColor('#0F172A'))
    c.setFont('Helvetica', 9.5)
    c.drawString(645, demo_y + 31, 'Full-Field Digital Mammography')
    if is_study_flagged:
        c.setFillColor(colors.HexColor('#B91C1C'))
        c.setFont('Helvetica-Bold', 10)
        c.drawString(645, demo_y + 7, 'ELEVATED MALIGNANCY RISK')
    else:
        c.setFillColor(colors.HexColor('#15803D'))
        c.setFont('Helvetica-Bold', 10)
        c.drawString(645, demo_y + 7, 'NEGATIVE / BENIGN')

    # Malignancy Cards
    cards_y, cards_h = H - 188, 66
    c_w = (W - 50 - 20) / 3.0

    # Left Breast Card
    c1_x = 25
    bg1 = colors.HexColor('#FFF1F2') if l_flag else colors.HexColor('#F8FAFC')
    bc1 = colors.HexColor('#E11D48') if l_flag else colors.HexColor('#CBD5E1')
    c.setFillColor(bg1)
    c.setStrokeColor(bc1)
    c.setLineWidth(1.5 if l_flag else 1)
    c.roundRect(c1_x, cards_y, c_w, cards_h, 4, fill=1, stroke=1)
    c.setFillColor(colors.HexColor('#FFE4E6') if l_flag else colors.HexColor('#E2E8F0'))
    c.roundRect(c1_x, cards_y + cards_h - 18, c_w, 18, 4, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#9F1239') if l_flag else colors.HexColor('#1E293B'))
    c.setFont('Helvetica-Bold', 9)
    c.drawString(c1_x + 10, cards_y + cards_h - 13, f"LEFT BREAST (L) - {'ELEVATED RISK' if l_flag else 'LOW RISK'}")

    pill_w, pill_h = 120, 18
    c.setFillColor(colors.HexColor('#FEE2E2') if l_flag else colors.HexColor('#DCFCE7'))
    c.setStrokeColor(colors.HexColor('#DC2626') if l_flag else colors.HexColor('#16A34A'))
    c.setLineWidth(1)
    c.roundRect(c1_x + 10, cards_y + 24, pill_w, pill_h, 3, fill=1, stroke=1)
    c.setFillColor(colors.HexColor('#991B1B') if l_flag else colors.HexColor('#166534'))
    c.setFont('Helvetica-Bold', 9)
    c.drawCentredString(c1_x + 10 + pill_w / 2.0, cards_y + 29, f"Malignancy: {l_prob_str}")
    c.setFillColor(colors.HexColor('#0F172A'))
    c.setFont('Helvetica', 8.5)
    c.drawString(c1_x + 10, cards_y + 9, f"Breast Density: ACR {l_dens}  |  Finding: {l_find}")
    if l_flag:
        c.setFillColor(colors.HexColor('#B91C1C'))
        c.setFont('Helvetica-Bold', 8.5)
        c.drawRightString(c1_x + c_w - 10, cards_y + 29, 'FLAGGED (>5%)')

    # Right Breast Card
    c2_x = 25 + c_w + 10
    bg2 = colors.HexColor('#FFF1F2') if r_flag else colors.HexColor('#F8FAFC')
    bc2 = colors.HexColor('#E11D48') if r_flag else colors.HexColor('#CBD5E1')
    c.setFillColor(bg2)
    c.setStrokeColor(bc2)
    c.setLineWidth(1.5 if r_flag else 1)
    c.roundRect(c2_x, cards_y, c_w, cards_h, 4, fill=1, stroke=1)
    c.setFillColor(colors.HexColor('#FFE4E6') if r_flag else colors.HexColor('#E2E8F0'))
    c.roundRect(c2_x, cards_y + cards_h - 18, c_w, 18, 4, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#9F1239') if r_flag else colors.HexColor('#1E293B'))
    c.setFont('Helvetica-Bold', 9)
    c.drawString(c2_x + 10, cards_y + cards_h - 13, f"RIGHT BREAST (R) - {'ELEVATED RISK' if r_flag else 'LOW RISK'}")

    c.setFillColor(colors.HexColor('#FEE2E2') if r_flag else colors.HexColor('#DCFCE7'))
    c.setStrokeColor(colors.HexColor('#DC2626') if r_flag else colors.HexColor('#16A34A'))
    c.setLineWidth(1)
    c.roundRect(c2_x + 10, cards_y + 24, pill_w, pill_h, 3, fill=1, stroke=1)
    c.setFillColor(colors.HexColor('#991B1B') if r_flag else colors.HexColor('#166534'))
    c.setFont('Helvetica-Bold', 9)
    c.drawCentredString(c2_x + 10 + pill_w / 2.0, cards_y + 29, f"Malignancy: {r_prob_str}")
    c.setFillColor(colors.HexColor('#0F172A'))
    c.setFont('Helvetica', 8.5)
    c.drawString(c2_x + 10, cards_y + 9, f"Breast Density: ACR {r_dens}  |  Finding: {r_find}")
    if r_flag:
        c.setFillColor(colors.HexColor('#B91C1C'))
        c.setFont('Helvetica-Bold', 8.5)
        c.drawRightString(c2_x + c_w - 10, cards_y + 29, 'FLAGGED (>5%)')

    # Card 3: Clinical Triage
    c3_x = 25 + 2 * (c_w + 10)
    c.setFillColor(colors.HexColor('#F8FAFC'))
    c.setStrokeColor(colors.HexColor('#CBD5E1'))
    c.setLineWidth(1)
    c.roundRect(c3_x, cards_y, c_w, cards_h, 4, fill=1, stroke=1)
    c.setFillColor(colors.HexColor('#E2E8F0'))
    c.roundRect(c3_x, cards_y + cards_h - 18, c_w, 18, 4, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#1E293B'))
    c.setFont('Helvetica-Bold', 9)
    c.drawString(c3_x + 10, cards_y + cards_h - 13, 'CLINICAL TRIAGE & RECOMMENDATION')

    elevated_info = []
    if l_flag: elevated_info.append(f"Left ({l_prob_str})")
    if r_flag: elevated_info.append(f"Right ({r_prob_str})")
    if elevated_info:
        c.setFillColor(colors.HexColor('#B91C1C'))
        c.setFont('Helvetica-Bold', 8.5)
        c.drawString(c3_x + 10, cards_y + 32, f"Elevated Risk: {', '.join(elevated_info)}")
    else:
        c.setFillColor(colors.HexColor('#15803D'))
        c.setFont('Helvetica-Bold', 8.5)
        c.drawString(c3_x + 10, cards_y + 32, 'Routine screening: Low probability')

    c.setFillColor(colors.HexColor('#475569'))
    c.setFont('Helvetica', 8)
    c.drawString(c3_x + 10, cards_y + 19, 'Action: Radiologic & clinical correlation advised.')
    c.drawString(c3_x + 10, cards_y + 7, f"Duplication: {patient_rec.get('single_view', 'None')}")

    # The 4 Views Section
    views_y = 26
    views_h = cards_y - views_y - 10
    v_gap = 14
    v_w = (W - 50 - 3 * v_gap) / 4.0

    four_views = [
        ('RIGHT CC (RCC)', imgs.get('R_CC'), r_prob_str, r_flag, 'Right Breast - Cranio-Caudal View'),
        ('LEFT CC (LCC)', imgs.get('L_CC'), l_prob_str, l_flag, 'Left Breast - Cranio-Caudal View'),
        ('RIGHT MLO (RMLO)', imgs.get('R_MLO'), r_prob_str, r_flag, 'Right Breast - Medio-Lateral Oblique View'),
        ('LEFT MLO (LMLO)', imgs.get('L_MLO'), l_prob_str, l_flag, 'Left Breast - Medio-Lateral Oblique View'),
    ]

    for idx, (v_title, img_p, p_score, is_v_flag, v_sub) in enumerate(four_views):
        vx = 25 + idx * (v_w + v_gap)
        hdr_h = 24
        hdr_bg = colors.HexColor('#FEE2E2') if is_v_flag else colors.HexColor('#0F2942')
        hdr_fg = colors.HexColor('#991B1B') if is_v_flag else colors.white
        border_c = colors.HexColor('#DC2626') if is_v_flag else colors.HexColor('#334155')

        c.setFillColor(colors.HexColor('#050505'))
        c.setStrokeColor(border_c)
        c.setLineWidth(2 if is_v_flag else 1)
        c.roundRect(vx, views_y, v_w, views_h, 4, fill=1, stroke=1)

        c.setFillColor(hdr_bg)
        c.rect(vx, views_y + views_h - hdr_h, v_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(hdr_fg)
        c.setFont('Helvetica-Bold', 9.5)
        c.drawString(vx + 8, views_y + views_h - 16, v_title)
        c.setFont('Helvetica-Bold', 8.5)
        c.drawRightString(vx + v_w - 8, views_y + views_h - 16, f"Risk: {p_score}")

        c.setFillColor(colors.HexColor('#1E293B'))
        c.rect(vx, views_y, v_w, 16, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#94A3B8'))
        c.setFont('Helvetica', 7.5)
        c.drawCentredString(vx + v_w / 2.0, views_y + 4.5, v_sub)

        if img_p and os.path.exists(img_p):
            im_area_x = vx + 4
            im_area_y = views_y + 18
            im_area_w = v_w - 8
            im_area_h = views_h - hdr_h - 20
            c.drawImage(img_p, im_area_x, im_area_y, width=im_area_w, height=im_area_h,
                        preserveAspectRatio=True, anchor='c')

    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont('Helvetica', 7.5)
    c.drawString(25, 10, 'CONFIDENTIAL MEDICAL REPORT  |  Full-Field Digital Mammography Bilateral Study  |  Page 1 of 3 (Study Overview)')
    c.drawRightString(W - 25, 10, f"Patient: {name} ({pid})  |  Date: {s_date} {s_time}")

    # ---------------- PAGE 2: HIGH-RES BILATERAL CC VIEWS ----------------
    c.showPage()
    hdr_h_page = 48
    c.setFillColor(colors.HexColor('#0F2942'))
    c.rect(0, H - hdr_h_page, W, hdr_h_page, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 13.5)
    c.drawString(25, H - 22, 'BILATERAL CRANIO-CAUDAL (CC) COMPARATIVE REVIEW')
    c.setFont('Helvetica', 9)
    c.drawString(25, H - 38, f"Patient: {name}  |  ID: {pid}  |  Date: {s_date}  |  Time: {s_time}  |  Age: {age} Y")
    c.setFont('Helvetica-Bold', 9.5)
    c.drawRightString(W - 25, H - 22, 'HIGH-RESOLUTION RADIOLOGICAL COMPARISON')

    cc_gap = 20
    cc_w = (W - 50 - cc_gap) / 2.0
    cc_y = 25
    cc_h = H - hdr_h_page - 38

    cc_views = [
        ('RIGHT CRANIO-CAUDAL (RCC)', imgs.get('R_CC'), r_prob_str, r_flag, f"Right Breast  |  Density: ACR {r_dens}  |  Finding: {r_find}"),
        ('LEFT CRANIO-CAUDAL (LCC)', imgs.get('L_CC'), l_prob_str, l_flag, f"Left Breast  |  Density: ACR {l_dens}  |  Finding: {l_find}"),
    ]

    for idx, (v_title, img_p, p_score, is_v_flag, v_sub) in enumerate(cc_views):
        cx = 25 + idx * (cc_w + cc_gap)
        hdr_h = 28
        hdr_bg = colors.HexColor('#FEE2E2') if is_v_flag else colors.HexColor('#0F2942')
        hdr_fg = colors.HexColor('#991B1B') if is_v_flag else colors.white
        border_c = colors.HexColor('#DC2626') if is_v_flag else colors.HexColor('#334155')

        c.setFillColor(colors.HexColor('#050505'))
        c.setStrokeColor(border_c)
        c.setLineWidth(2 if is_v_flag else 1)
        c.roundRect(cx, cc_y, cc_w, cc_h, 4, fill=1, stroke=1)

        c.setFillColor(hdr_bg)
        c.rect(cx, cc_y + cc_h - hdr_h, cc_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(hdr_fg)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(cx + 12, cc_y + cc_h - 19, v_title)
        c.setFont('Helvetica-Bold', 10)
        status_txt = f"Malignancy Risk: {p_score}  [FLAGGED > 5%]" if is_v_flag else f"Malignancy Risk: {p_score}  [Normal]"
        c.drawRightString(cx + cc_w - 12, cc_y + cc_h - 19, status_txt)

        c.setFillColor(colors.HexColor('#1E293B'))
        c.rect(cx, cc_y, cc_w, 20, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#94A3B8'))
        c.setFont('Helvetica', 8.5)
        c.drawCentredString(cx + cc_w / 2.0, cc_y + 6, v_sub)

        if img_p and os.path.exists(img_p):
            im_area_x = cx + 6
            im_area_y = cc_y + 24
            im_area_w = cc_w - 12
            im_area_h = cc_h - hdr_h - 28
            c.drawImage(img_p, im_area_x, im_area_y, width=im_area_w, height=im_area_h,
                        preserveAspectRatio=True, anchor='c')

    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont('Helvetica', 7.5)
    c.drawString(25, 10, 'CONFIDENTIAL MEDICAL REPORT  |  Full-Field Digital Mammography Bilateral Study  |  Page 2 of 3 (CC Views)')
    c.drawRightString(W - 25, 10, f"Patient: {name} ({pid})  |  Date: {s_date} {s_time}")

    # ---------------- PAGE 3: HIGH-RES BILATERAL MLO VIEWS ----------------
    c.showPage()
    c.setFillColor(colors.HexColor('#0F2942'))
    c.rect(0, H - hdr_h_page, W, hdr_h_page, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 13.5)
    c.drawString(25, H - 22, 'BILATERAL MEDIO-LATERAL OBLIQUE (MLO) COMPARATIVE REVIEW')
    c.setFont('Helvetica', 9)
    c.drawString(25, H - 38, f"Patient: {name}  |  ID: {pid}  |  Date: {s_date}  |  Time: {s_time}  |  Age: {age} Y")
    c.setFont('Helvetica-Bold', 9.5)
    c.drawRightString(W - 25, H - 22, 'HIGH-RESOLUTION RADIOLOGICAL COMPARISON')

    mlo_views = [
        ('RIGHT MEDIO-LATERAL OBLIQUE (RMLO)', imgs.get('R_MLO'), r_prob_str, r_flag, f"Right Breast  |  Density: ACR {r_dens}  |  Finding: {r_find}"),
        ('LEFT MEDIO-LATERAL OBLIQUE (LMLO)', imgs.get('L_MLO'), l_prob_str, l_flag, f"Left Breast  |  Density: ACR {l_dens}  |  Finding: {l_find}"),
    ]

    for idx, (v_title, img_p, p_score, is_v_flag, v_sub) in enumerate(mlo_views):
        cx = 25 + idx * (cc_w + cc_gap)
        hdr_h = 28
        hdr_bg = colors.HexColor('#FEE2E2') if is_v_flag else colors.HexColor('#0F2942')
        hdr_fg = colors.HexColor('#991B1B') if is_v_flag else colors.white
        border_c = colors.HexColor('#DC2626') if is_v_flag else colors.HexColor('#334155')

        c.setFillColor(colors.HexColor('#050505'))
        c.setStrokeColor(border_c)
        c.setLineWidth(2 if is_v_flag else 1)
        c.roundRect(cx, cc_y, cc_w, cc_h, 4, fill=1, stroke=1)

        c.setFillColor(hdr_bg)
        c.rect(cx, cc_y + cc_h - hdr_h, cc_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(hdr_fg)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(cx + 12, cc_y + cc_h - 19, v_title)
        c.setFont('Helvetica-Bold', 10)
        status_txt = f"Malignancy Risk: {p_score}  [FLAGGED > 5%]" if is_v_flag else f"Malignancy Risk: {p_score}  [Normal]"
        c.drawRightString(cx + cc_w - 12, cc_y + cc_h - 19, status_txt)

        c.setFillColor(colors.HexColor('#1E293B'))
        c.rect(cx, cc_y, cc_w, 20, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#94A3B8'))
        c.setFont('Helvetica', 8.5)
        c.drawCentredString(cx + cc_w / 2.0, cc_y + 6, v_sub)

        if img_p and os.path.exists(img_p):
            im_area_x = cx + 6
            im_area_y = cc_y + 24
            im_area_w = cc_w - 12
            im_area_h = cc_h - hdr_h - 28
            c.drawImage(img_p, im_area_x, im_area_y, width=im_area_w, height=im_area_h,
                        preserveAspectRatio=True, anchor='c')

    c.setFillColor(colors.HexColor('#64748B'))
    c.setFont('Helvetica', 7.5)
    c.drawString(25, 10, 'CONFIDENTIAL MEDICAL REPORT  |  Full-Field Digital Mammography Bilateral Study  |  Page 3 of 3 (MLO Views)')
    c.drawRightString(W - 25, 10, f"Patient: {name} ({pid})  |  Date: {s_date} {s_time}")

    c.save()


def generate_flagged_patient_reports(output_dir, study_records):
    if not HAS_REPORTLAB:
        print("[PDF] reportlab is not available; skipping PDF report generation.")
        return

    flagged_records = [r for r in study_records if r.get('flagged', False)]
    if not flagged_records:
        print("\n[PDF] No flagged patients (>5% malignancy risk) found in this batch.")
        return

    print(f"\n[PDF] Generating Professional Reports for {len(flagged_records)} Flagged Patient(s)...")
    flagged_dir = os.path.join(output_dir, 'flagged_patients_reports')
    os.makedirs(flagged_dir, exist_ok=True)

    generated_pdfs = []
    for idx, rec in enumerate(flagged_records, 1):
        pid = rec['patient_id']
        name = rec['patient_name']
        print(f"  [{idx}/{len(flagged_records)}] Generating report for: {pid} ({name})")

        # 1. In individual patient directory
        pdir = os.path.join(output_dir, pid)
        os.makedirs(pdir, exist_ok=True)
        patient_pdf_path = os.path.join(pdir, f"{pid}_report.pdf")
        render_high_res_patient_pdf(patient_pdf_path, rec, pdir)

        # 2. Copy to central flagged reports folder
        flagged_pdf_path = os.path.join(flagged_dir, f"{pid}_report.pdf")
        shutil.copyfile(patient_pdf_path, flagged_pdf_path)

        # 3. Copy to root output folder as latest export
        root_pdf_path = os.path.join(output_dir, f"{pid}_report.pdf")
        shutil.copyfile(patient_pdf_path, root_pdf_path)

        generated_pdfs.append(patient_pdf_path)

    # 3. Consolidated Multi-Patient PDF
    if generated_pdfs:
        print("  -> Merging flagged reports into consolidated PDF...")
        writer = PdfWriter()
        for pdf in generated_pdfs:
            writer.append(pdf)

        consolidated_pdf = os.path.join(flagged_dir, "all_flagged_patients_consolidated.pdf")
        root_consolidated_pdf = os.path.join(output_dir, "all_flagged_patients_consolidated.pdf")

        with open(consolidated_pdf, 'wb') as f_out:
            writer.write(f_out)
        writer.close()
        shutil.copyfile(consolidated_pdf, root_consolidated_pdf)
        print(f"  [OK] Consolidated PDF generated: {consolidated_pdf}")


# =====================================================================
# 8. MAIN BATCH PROCESSING PIPELINE
# =====================================================================

def process_batch(input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR,
                  start_date=None, end_date=None, organize_input=False,
                  device_str=None, skip_pdf=False):

    print("=" * 72)
    print("MAMMOGRAPHY BATCH PROCESSOR & REPORTING PIPELINE (V4)")
    print("=" * 72)
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")
    if start_date or end_date:
        print(f"Date Filter      : {start_date or 'Beginning'} to {end_date or 'Present'}")
    if organize_input:
        print(f"Input Organizing : Enabled (raw DICOMs will be copied into subfolders)")

    os.makedirs(output_dir, exist_ok=True)

    # Discover and segregate studies
    studies = discover_dicom_studies(input_dir, start_date=start_date, end_date=end_date,
                                     organize_input=organize_input)

    if not studies:
        print(f"\n[INFO] No matching patient studies found to process in: {input_dir}")
        return

    print(f"\n[FOUND] {len(studies)} patient study bundle(s) ready for AI inference.\n")

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using computing device: {device}")

    model = load_mammo_model(device)
    study_records = []
    summary_excel_path = os.path.join(output_dir, 'mammography_summary.xlsx')

    for idx, study in enumerate(studies, 1):
        folder_name = study['folder_name']
        patient_id = folder_name
        p_name = study['patient_name']
        p_age = study['patient_age']
        p_date = study['scanned_date']
        p_time = study['scanned_time']
        dcm_files = study['dcm_files']

        print(f"==================================================")
        print(f"[{idx}/{len(studies)}] Processing Study: {folder_name}")
        print(f"  -> Demographics: Name='{p_name or 'N/A'}' | Age={p_age} | Date={p_date} {p_time}")

        patient_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(patient_output_dir, exist_ok=True)

        left_breast = {'cc': None, 'mlo': None}
        right_breast = {'cc': None, 'mlo': None}
        internal_pids_seen = set()
        failed_files = []
        saved_png_views = {}

        for file_path in dcm_files:
            filename = os.path.basename(file_path)
            internal_pid, lat, view = get_dicom_view_metadata(file_path)
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
            png_out_path = os.path.join(patient_output_dir, f"{base_name}_{lat}_{view}.png")
            Image.fromarray(img_array).save(png_out_path)

            if view != 'UNK' and lat != 'UNK':
                view_key = f"{lat}_{view}"
                if '_Ann_' not in filename:
                    saved_png_views[view_key] = png_out_path

            if view == 'UNK' or lat == 'UNK':
                print(f"     [NOTE] {filename} saved but excluded from AI ensemble (side/view undetermined)")
                continue

            view_key = 'mlo' if view == 'MLO' else 'cc'
            if lat == 'L':
                left_breast[view_key] = img_array
            else:
                right_breast[view_key] = img_array

        log_id_check(folder_name, internal_pids_seen)

        # Single-view duplication fallback
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
            print(f"  [SKIP] No usable CC/MLO pair for {folder_name} -- skipping inference.")
            continue

        with torch.inference_mode():
            output = model(inference_payloads, device=device)

        txt_filepath = os.path.join(patient_output_dir, f"Study_Report_{patient_id}.txt")
        density_classes = ["A (Fatty)", "B (Scattered)", "C (Heterogeneous)", "D (Dense)"]

        left_prob, left_density, left_class = None, None, None
        right_prob, right_density, right_class = None, None, None

        with open(txt_filepath, 'w') as f:
            f.write("=== CLINICAL AI MAMMOGRAPHY STUDY REPORT ===\n")
            f.write(f"Patient ID   : {patient_id}\n")
            f.write(f"Folder       : {folder_name}\n")
            if p_name:
                f.write(f"Patient Name : {p_name}\n")
            if p_age != "N/A":
                f.write(f"Age          : {p_age}\n")
            if p_date != "Unknown Date":
                f.write(f"Scanned Date : {p_date}\n")
            if p_time != "Unknown Time":
                f.write(f"Scanned Time : {p_time}\n")
            if failed_files:
                f.write(f"WARNING      : {len(failed_files)} file(s) failed to process: {', '.join(failed_files)}\n")
            f.write("--------------------------------------------\n\n")

            for p_idx, side_name in enumerate(side_labels):
                cancer_prob = output['cancer'][p_idx].item() * 100
                cancer_class = "Suspicious/Malignant" if cancer_prob > 50 else "Negative/Benign"
                density_idx = output['density'][p_idx].argmax(0).item()
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

                side_word = side_name.split()[0]
                if side_word in duplicated_sides:
                    f.write("NOTE: Only a single view was available for this side. The second view\n")
                    f.write("was duplicated to satisfy model input requirement (lower confidence).\n")
                f.write("--------------------------------------------\n\n")

        print(f"  [SUCCESS] Results saved: output/{folder_name}/")

        is_flagged = (left_prob is not None and left_prob >= 5.0) or (right_prob is not None and right_prob >= 5.0)

        study_records.append({
            "scanned_date": p_date,
            "scanned_time": p_time,
            "patient_id": patient_id,
            "patient_name": p_name,
            "patient_age": p_age,
            "left_prob": left_prob,
            "left_density": left_density,
            "left_finding": left_class,
            "right_prob": right_prob,
            "right_density": right_density,
            "right_finding": right_class,
            "single_view": ", ".join(duplicated_sides) if duplicated_sides else "None",
            "flagged": is_flagged,
            "notes": f"{len(failed_files)} file(s) failed" if failed_files else "OK",
            "image_paths": saved_png_views
        })

    # Export Excel Summary
    if study_records:
        export_to_excel(study_records, summary_excel_path)
        generate_summary_pdf(output_dir)

    # Generate High-Resolution 4-View PDF Reports for Flagged Patients
    if not skip_pdf and study_records:
        generate_flagged_patient_reports(output_dir, study_records)

    print("\n" + "=" * 72)
    print("ALL MAMMOGRAPHY PROCESSING & REPORTING TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 72)


# =====================================================================
# 9. CLI ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mammography DICOM Batch Processor & Clinical Reporter (v4)"
    )
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR,
                        help=f"Path to input DICOM directory (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help=f"Path to output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument('--start-date', default=None,
                        help="Filter studies on or after this date (YYYY-MM-DD or YYYYMMDD)")
    parser.add_argument('--end-date', default=None,
                        help="Filter studies on or before this date (YYYY-MM-DD or YYYYMMDD)")
    parser.add_argument('--organize-input', action='store_true',
                        help="Physically copy/move raw DICOMs into segregated subfolders in input-dir")
    parser.add_argument('--device', default=None,
                        help="Device to run inference on: 'cuda' or 'cpu'")
    parser.add_argument('--skip-pdf', action='store_true',
                        help="Skip generating individual flagged patient PDF reports")

    args = parser.parse_args()

    process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        organize_input=args.organize_input,
        device_str=args.device,
        skip_pdf=args.skip_pdf
    )


if __name__ == "__main__":
    main()

# Mammography DICOM Image Processor

A local-first clinical AI screening pipeline for digital mammography built on
[Ian Pan's `mammoscreen` model](https://huggingface.co/ianpan/mammoscreen).
It reads raw DICOM mammography studies and produces, for each breast:

- **Malignancy probability** — a percentage score reflecting cancer likelihood
- **ACR breast density classification** (A–D) — flags dense tissue that can
  mask tumors and cause false negatives

## Why this model

- **Dual clinical utility** — most open-source medical models only output a
  binary Normal/Abnormal prediction; this one outputs cancer risk *and*
  density in a single pass.
- **High-quality training data** — trained and validated on the global RSNA
  Screening Mammography Challenge dataset (54,000+ images), so it is optimized
  for subtle, early-stage signs in asymptomatic populations rather than
  diagnostic (palpable-lump) cases.
- **Native DICOM handling** — the model's built-in `load_image_from_dicom`
  method handles 12/16-bit pixel depths and manufacturer VOI LUTs (windowing,
  contrast-leveling, normalization) internally, so raw files never come out
  black or blown out.
- **Lightweight & offline** — ~244 MB weights run on a standard laptop CPU in
  seconds, with no GPU or cloud dependency. Once weights are cached locally the
  pipeline runs fully offline — ideal for rural screening programs, mobile
  medical camps, and satellite clinics.

## Features

- Batch processing of patient study folders (`input/<patient-id>/*.dcm`)
- CC/MLO view and left/right laterality detection from DICOM metadata, with
  safe `UNK` handling (unknown views are still saved as PNGs but excluded from
  the AI ensemble rather than silently mislabeled)
- Deterministic OpenCV breast ROI cropping that strips background dead space,
  noise, and burned-in text labels
- Automatic view duplication for single-view / incomplete studies, with a
  lower-confidence note written into the report
- Per-patient text study reports plus per-view PNG previews
- Privacy-first design: only the anonymized folder name is used in any output;
  DICOM-internal `PatientID` is used solely for an audit-log cross-check that
  catches files copied into the wrong patient folder

## Scripts

| Script | Description |
| --- | --- |
| `single-dicom-processor.py` | Process one DICOM file dropped into `input/` |
| `batch-dicom-processor.py` | Original batch version (per-patient folders) |
| `batch-dicom-processor-V2.py` | Adds RGB flattening + view duplication fix |
| `batch-dicom-processor-v3.py` | **Recommended.** Margin/coordinate fix in ROI cropping, `UNK` view & laterality safety, failed-file tracking in reports, per-folder audit log |

## Requirements

- Python 3.8+
- PyTorch
- `transformers`
- `pydicom`
- `opencv-python`
- `numpy`
- `Pillow`

Install with:

```bash
pip install torch transformers pydicom opencv-python numpy Pillow
```

## Usage

### Single file

Place a `.dcm` file in `input/` and run:

```bash
python single-dicom-processor.py
```

### Batch (recommended)

1. Create one subfolder per patient under `input/` using an **anonymized**
   identifier (e.g. `input/28-n/`), and place that patient's DICOM files inside.
2. Run:

```bash
python batch-dicom-processor-v3.py
```

The first run downloads the model weights from the Hugging Face Hub (~244 MB)
and caches them locally; subsequent runs work offline. For strict offline
deployment, add `local_files_only=True` to the `AutoModel.from_pretrained(...)`
call in the script.

## Input structure

```
input/
└── <anonymized-patient-id>/
    ├── study1.dcm
    ├── study2.dcm
    └── ...
```

## Output structure

```
output/
└── <anonymized-patient-id>/
    ├── Study_Report_<patient-id>.txt
    ├── <file>_L_CC.png
    └── <file>_R_MLO.png
```

Example report:

```
=== CLINICAL AI MAMMOGRAPHY STUDY REPORT ===
Patient ID : test
Folder     : test
--------------------------------------------

--- LEFT BREAST (L) ---
Cancer Probability : 96.0% (Suspicious/Malignant)
Breast Density     : B (Scattered)
NOTE: Only a single view was available for this side. The second view
was duplicated to satisfy the model's two-view input requirement.
Treat this result as lower-confidence than a true two-view study.
--------------------------------------------
```

## How it works

1. **Read** — each DICOM file is parsed with the model's built-in DICOM loader
   (windowing / VOI LUT / normalization handled natively).
2. **Normalize** — the pixel array is min-max normalized to 8-bit.
3. **Crop** — `crop_breast_roi` isolates the largest connected tissue region,
   removing black padding and label burns.
4. **Bucket** — images are sorted into CC/MLO slots per breast based on DICOM
   metadata; unknown views are excluded from inference.
5. **Infer** — each breast's two-view pair is passed through the ensemble to
   get cancer probability and density class.
6. **Report** — a per-patient text report and PNG previews are written to
   `output/`.

## Privacy & data handling

- Patient folders under `input/` must already be anonymized; the folder name is
  the only identifier used in outputs.
- DICOM-internal `PatientID` values are never written to reports or filenames.
  They are only counted (distinct IDs per folder) in `audit_log.txt` to trip a
  warning if files from different patients end up in the same folder.
- `input/`, `output/`, `model_weights/`, and `audit_log.txt` are excluded from
  version control — never commit raw medical imaging data.

## Disclaimer

This tool is a research/screening-support pipeline, not a certified medical
device. Results must be reviewed by a qualified radiologist and are not a
substitute for professional diagnosis.

## License

[MIT](LICENSE)

# Mammography DICOM Batch Processor

A **local-first clinical AI screening pipeline** for digital mammography. It
processes whole patient study folders of raw DICOM images and, for each breast,
produces:

- **Malignancy probability** — a percentage score reflecting the likelihood of
  cancerous tissue
- **ACR breast density classification** (A–D) — flags dense tissue that can
  physically mask tumors and cause false negatives

The AI engine is Dr. Ian Pan's open-source **`mammoscreen`** model — see
[Credits & Acknowledgments](#credits--acknowledgments) below.

---

## Why this model

- **Dual clinical utility** — most open-source medical models only output a
  binary Normal/Abnormal prediction. `mammoscreen` simultaneously outputs
  malignancy probability *and* ACR density class (A–D) in a single pass.
- **High-quality training data** — trained and validated on the global
  **RSNA Screening Mammography Challenge** dataset (54,000+ images).
  Because it was trained on *screening* data (asymptomatic populations)
  rather than diagnostic data (patients with palpable lumps), it is optimized
  to catch subtle, early-stage signs and avoids the false-positive bias common
  in diagnostic-only models.
- **Native DICOM handling** — raw DICOM pixel arrays are often 12/16-bit with
  manufacturer-specific VOI LUTs, which makes them look black or blown out to
  standard computer-vision scripts. The model's built-in
  `load_image_from_dicom` handles medical windowing, contrast-leveling, and
  normalization internally — no error-prone custom math needed.
- **Lightweight & offline-first** — the compiled weights are ~244 MB and run on
  a standard laptop CPU in seconds, with **no GPU or cloud dependency**. Once
  the weights are cached locally, the script runs fully offline — ideal for
  rural screening programs, mobile medical camps, and satellite clinics.

---

## Features

- **Batch processing** of patient study folders (`input/<patient-id>/*.dcm`)
- **View/laterality detection** — CC/MLO and left/right are read from DICOM
  metadata (`ViewPosition`, `ImageLaterality`, `SeriesDescription`), with safe
  `UNK` handling: unknown views are still saved as PNG previews but excluded
  from the AI ensemble rather than silently mislabeled
- **Deterministic breast ROI cropping** — OpenCV connected-component analysis
  strips black padding, noise, and burned-in text labels to isolate tissue
- **Single-view fallback** — when only one view (CC or MLO) is available, the
  missing view is duplicated to satisfy the model's two-view input, and the
  report explicitly flags the result as **lower confidence**
- **Per-patient text reports** + **per-view PNG previews**
- **Failure tracking** — files that fail to decode are listed in the report
  instead of being silently dropped
- **Privacy-first** — only the anonymized folder name is used in any output;
  DICOM-internal `PatientID` values are used *only* for an audit-log
  cross-check that catches files copied into the wrong patient folder

---

## Requirements

- Python 3.8+
- PyTorch
- `transformers`
- `pydicom`
- `opencv-python`
- `numpy`
- `Pillow`

```bash
pip install torch transformers pydicom opencv-python numpy Pillow
```

---

## Usage

### 1. Organize input

Create one subfolder per patient under `input/`, named with an **anonymized**
identifier, and place that patient's DICOM files inside:

```
input/
└── 28-n/
    ├── study_01.dcm
    ├── study_02.dcm
    └── ...
```

### 2. Run

```bash
python batch-dicom-processor-v3.py
```

On the **first run**, the script downloads the model weights (~244 MB) from the
Hugging Face Hub and caches them locally; all subsequent runs work **fully
offline**. For strict offline deployment (e.g. a mobile van), add
`local_files_only=True` to the `AutoModel.from_pretrained(...)` call.

The script automatically uses a CUDA GPU when available and falls back to CPU.

---

## Output

```
output/
└── 28-n/
    ├── Study_Report_28-n.txt
    ├── study_01_L_CC.png
    └── study_02_R_MLO.png
```

Example report:

```
=== CLINICAL AI MAMMOGRAPHY STUDY REPORT ===
Patient ID : 28-n
Folder     : 28-n
--------------------------------------------

--- LEFT BREAST (L) ---
Cancer Probability : 96.0% (Suspicious/Malignant)
Breast Density     : B (Scattered)
NOTE: Only a single view was available for this side. The second view
was duplicated to satisfy the model's two-view input requirement.
Treat this result as lower-confidence than a true two-view study.
--------------------------------------------
```

A `WARNING` line is added to the report when some DICOM files failed to
process, and a per-folder entry is appended to `audit_log.txt` cross-checking
the folder name against the internal `PatientID` tags found in its files.

---

## How it works

1. **Read** — each DICOM is parsed with the model's built-in loader
   (windowing / VOI LUT / normalization handled natively).
2. **Normalize** — pixel data is min-max normalized to 8-bit grayscale.
3. **Crop** — `crop_breast_roi` isolates the largest connected tissue region,
   removing background dead space and label burns.
4. **Bucket** — images are sorted into CC/MLO slots per breast from DICOM
   metadata; unknown views are excluded from inference.
5. **Complete** — missing views are duplicated (and flagged) so every breast
   has the two-view pair the ensemble expects.
6. **Infer** — each breast's two-view pair is passed through the model to get
   cancer probability and density class.
7. **Report** — a per-patient text report and PNG previews are written to
   `output/`.

---

## Credits & Acknowledgments

This project would not exist without the open work of **Dr. Ian Pan**,
radiologist and deep-learning practitioner, who published the `mammoscreen`
model openly on the Hugging Face Hub:

| | |
| --- | --- |
| **Personal website** | [https://ianpan.me](https://ianpan.me) |
| **GitHub** | [https://i-pan.github.io](https://i-pan.github.io) |
| **Hugging Face profile** | [https://huggingface.co/ianpan](https://huggingface.co/ianpan) |
| **Model card** | [https://huggingface.co/ianpan/mammoscreen](https://huggingface.co/ianpan/mammoscreen) |

The `mammoscreen` model simultaneously outputs malignancy probability and ACR
breast density classification, was trained and validated on the RSNA Screening
Mammography Challenge dataset, and is used here under its original open-access
terms — please refer to the model card for the exact license and usage terms.

---

## Privacy & data handling

- Patient folders under `input/` must already be anonymized; the folder name is
  the only identifier used in outputs.
- DICOM-internal `PatientID` values are **never** written to reports or
  filenames. They are only counted in `audit_log.txt` (distinct IDs per folder)
  to trip a warning if files from different patients end up in the same folder.
- `input/`, `output/`, `model_weights/`, and `audit_log.txt` are excluded from
  version control — never commit raw medical imaging data.

---

## Disclaimer

This tool is a **research / screening-support pipeline, not a certified
medical device**. Results must be reviewed by a qualified radiologist and are
not a substitute for professional diagnosis.

---

## License

[MIT](LICENSE)

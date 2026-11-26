# Mammography DICOM Batch Processor

A **local-first clinical AI screening pipeline** for digital mammography. It
processes whole patient study folders of raw DICOM images and, for each breast,
produces:

- **Malignancy probability** — a percentage score reflecting the likelihood of
  cancerous tissue
- **ACR breast density classification** (A–D) — flags dense tissue that can
  physically mask tumors and cause false negatives

The AI engine is **Dr. Ian Pan's** open-source **`mammoscreen`** model
([website](https://ianpan.me/) · [Hugging Face model](https://huggingface.co/ianpan/mammoscreen)) —
see [Credits & Acknowledgments](#credits--acknowledgments) below.

---

## Why this model

- **Dual clinical utility** — most open-source medical models only output a
  binary Normal/Abnormal prediction. [`mammoscreen`](https://huggingface.co/ianpan/mammoscreen)
  simultaneously outputs malignancy probability *and* ACR density class (A–D)
  in a single pass.
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

- **Bulk DICOM ingestion** — drop a flat PACS/workstation export (or mixed/nested folder) into `input/` and the processor auto-segregates files into distinct patient studies by DICOM metadata (`StudyInstanceUID`, `PatientID`, `StudyDate`)
- **Backward-compatible folder mode** — pre-organized `input/<patient-id>/*.dcm` study folders still work exactly as before
- **View/laterality detection** — CC/MLO and left/right are read from DICOM
  metadata (`ViewPosition`, `ImageLaterality`, `SeriesDescription`), with safe
  `UNK` handling: unknown views are still saved as PNG previews but excluded
  from the AI ensemble rather than silently mislabeled
- **Deterministic breast ROI cropping** — OpenCV connected-component analysis
  strips black padding, noise, and burned-in text labels to isolate tissue
- **Single-view fallback** — when only one view (CC or MLO) is available, the
  missing view is duplicated to satisfy the model's two-view input, and the
  report explicitly flags the result as **lower confidence**
- **Consolidated Excel report** — automatically generates `output/mammography_summary.xlsx` aggregating all studies, with bilateral malignancy probabilities, separate breast density columns, patient age/name, and scanned dates clubbed together (plus individual tabs per scan date)
- **Per-patient text reports** + **per-view PNG previews**
- **Professional PDF reports** — flagged patients (>5% malignancy risk) get high-resolution 4-view clinical PDF reports, plus a consolidated flagged-patients PDF
- **CLI controls** — `--input-dir`, `--output-dir`, `--start-date`, `--end-date`, `--organize-input`, `--device`, `--skip-pdf`
- **Failure tracking** — files that fail to decode are listed in the report
  instead of being silently dropped
- **Privacy-first** — folder names are the authoritative study identifier;
  DICOM-internal `PatientID` values are used *only* for an audit-log
  cross-check that catches files copied into the wrong patient folder

---

## Requirements

- **Python 3.10 (recommended)** — newer Python versions may lack prebuilt wheels for some dependencies (e.g. `tokenizers`) and can cause install/build issues.
- **~500 MB free disk space** for dependencies and model weights
- **Operating system**: Windows, macOS, or Linux

> **Environment recommendation:** On Linux, prefer **`uv`** or **conda** over a plain `venv`. `uv`/conda make it easy to install and pin Python 3.10 and pull prebuilt wheels; plain `venv` uses whatever system Python is installed and can hit build/compiler issues on newer Python versions. On Windows, conda is the easiest option.

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `transformers` (pinned to 4.45.x) | Hugging Face model loading |
| `timm` | Pre-trained image model architectures |
| `albumentations` | Image augmentation library |
| `pydicom` | DICOM file parsing & metadata extraction |
| `opencv-python` | Image processing and breast ROI cropping |
| `numpy` | Numerical operations |
| `Pillow` | PNG image saving |
| `openpyxl` | Formatted Excel summary export |
| `reportlab` | Professional PDF report rendering |
| `pypdf` | Consolidated multi-patient PDF merging |

---

## Installation (Step-by-Step)

### Option A: Using `uv` (Recommended for Linux & macOS — Fast & Clean)

[`uv`](https://github.com/astral-sh/uv) is an extremely fast, standalone Python package and environment manager. It automatically downloads and manages a dedicated **Python 3.10** runtime in your user directory without needing root (`sudo`) access and without modifying your system Python. All dependencies install directly from pre-built wheels in seconds (no C/Rust compilers required).

#### Step 1: Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

#### Step 2: Create a Python 3.10 virtual environment

From the project root:

```bash
uv venv --python 3.10 mammo
```

#### Step 3: Activate the environment

```bash
source mammo/bin/activate
```

#### Step 4: Install dependencies

```bash
uv pip install -r requirements.txt
```

#### Step 5: Verify installation

```bash
python -c "import torch, transformers, timm, albumentations, openpyxl; print('All dependencies installed successfully!')"
```

---

### Option B: Using Conda (Recommended for Windows)

Conda is the easiest way to manage dependencies on Windows, where some packages (like `tokenizers`) require special build tools.

#### Step 1: Install Miniconda

Download and install Miniconda for your operating system:

- **Windows**: https://docs.conda.io/en/latest/miniconda.html#windows-installers
- **macOS**: https://docs.conda.io/en/latest/miniconda.html#macosx-installers
- **Linux**: https://docs.conda.io/en/latest/miniconda.html#linux-installers

After installation, open **Anaconda Prompt** (Windows) or **Terminal** (macOS/Linux).

#### Step 2: Create a dedicated environment

```bash
conda create -n mammo python=3.10 -y
```

#### Step 3: Activate the environment

```bash
conda activate mammo
```

> **Important**: Always activate this environment before running the script.
> You'll see `(mammo)` at the beginning of your command prompt when it's active.

#### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision transformers==4.45.2 timm albumentations pydicom opencv-python numpy Pillow openpyxl reportlab pypdf
```

> **Note**: `transformers` is pinned to version 4.45.2 for compatibility with
> the mammoscreen model. Newer versions (5.x) may cause errors.

#### Step 5: Verify installation

```bash
python -c "import torch, transformers, timm, albumentations, openpyxl; print('All dependencies installed successfully!')"
```

You should see: `All dependencies installed successfully!`

---

### Option C: Using standard `venv` & `pip` (Not recommended on Linux)

> Use this only if you already have **Python 3.10** installed. On Linux, prefer Option A (`uv`) or Option B (conda) to avoid system-Python and wheel/build issues.

If your system already has Python 3.10 installed:

```bash
# Create a virtual environment
python3.10 -m venv mammo

# Activate it
# Windows:
mammo\Scripts\activate
# macOS/Linux:
source mammo/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Option D: Using Docker (Advanced)

For reproducible environments across different machines:

```bash
docker run -it --rm -v $(pwd):/app -w /app python:3.10-slim bash
pip install -r requirements.txt
python batch-dicom-processor-v4.py --input-dir ./input --output-dir ./output
```

---

## Usage

### 1. Prepare input

**Option A — Bulk flat folder (v4, recommended for PACS exports):** place all DICOM
files directly in `input/` (or point `--input-dir` at any export folder). The
processor inspects DICOM headers and automatically groups files into distinct
patient studies.

```
input/
├── export_001.dcm
├── export_002.dcm
├── export_003.dcm
└── ...
```

**Option B — Pre-organized patient folders (backward compatible):** create one
subfolder per patient/study under `input/`:

```
input/
└── 28-n/
    ├── study_01.dcm
    ├── study_02.dcm
    └── ...
```

> **Tip**: Folder names should not contain any patient-identifying information.
> Use anonymized identifiers like `patient-001`, `study-A`, etc.

### 2. Run the processor

```bash
# Activate the environment
# conda:  conda activate mammo
# uv/venv: source mammo/bin/activate

# v4: bulk import (auto-segregates DICOMs by study metadata)
python batch-dicom-processor-v4.py --input-dir ./input --output-dir ./output

# Filter by scan date range (YYYY-MM-DD or YYYYMMDD)
python batch-dicom-processor-v4.py --start-date 2026-01-01 --end-date 2026-01-31

# Physically copy raw DICOMs into segregated input subfolders
python batch-dicom-processor-v4.py --organize-input

# Force CPU / skip PDF report generation
python batch-dicom-processor-v4.py --device cpu --skip-pdf
```

**v4 CLI options:**

| Flag | Description |
|------|-------------|
| `--input-dir` | Input DICOM directory (default: `./input`) |
| `--output-dir` | Output directory (default: `./output`) |
| `--start-date` | Include studies on/after this date (`YYYY-MM-DD` or `YYYYMMDD`) |
| `--end-date` | Include studies on/before this date (`YYYY-MM-DD` or `YYYYMMDD`) |
| `--organize-input` | Copy raw DICOMs into segregated subfolders inside input dir |
| `--device` | Inference device: `cuda` or `cpu` (default: auto-detect) |
| `--skip-pdf` | Skip flagged-patient PDF report generation |

> **v3 compatibility:** `batch-dicom-processor-v3.py` is still included for
> pre-organized per-patient folder workflows.

On the **first run**, the script downloads the model weights (~244 MB) from the
Hugging Face Hub and caches them locally; all subsequent runs work **fully
offline**. For strict offline deployment (e.g. a mobile van), add
`local_files_only=True` to the `AutoModel.from_pretrained(...)` call.

The script automatically uses a CUDA GPU when available and falls back to CPU.

### 3. Check results

Results are saved in the `output/` folder, organized by patient ID:

```
output/
├── mammography_summary.xlsx     # Consolidated Excel summary (all studies, clubbed by date)
└── 28-n/
    ├── Study_Report_28-n.txt    # AI analysis report (with Name, Age, Scanned Date)
    ├── study_01_L_CC.png        # Left breast, CC view
    └── study_02_R_MLO.png       # Right breast, MLO view
```

---

## Output

### 1. Consolidated Excel Summary (`output/mammography_summary.xlsx`)

At the conclusion of each batch run, an Excel workbook is automatically generated with all patient studies consolidated and clubbed by **Scanned Date**:

- **Master Sheet (`Mammography Summary`)**: Chronologically sorted by scanned date, separated with visual date group banners showing patient counts per date.
- **Per-Date Sheets**: When multiple distinct scan dates are present in the batch, individual date tabs (e.g. `2026-07-21`) are also generated for easy day-by-day clinical auditing.
- **Key Columns**:
  1. `Scanned Date` — Extracted from DICOM `StudyDate` (fallbacks: `SeriesDate`/`AcquisitionDate`/`ContentDate`)
  2. `Patient ID` — Study folder identifier
  3. `Patient Name` — DICOM `PatientName` tag (if present)
  4. `Age` — Patient age parsed from DICOM header (e.g. `58`), or calculated from birth date
  5. `Left Malignancy (%)` — Probability score for left breast
  6. `Left Breast Density` — ACR density class (A: Fatty, B: Scattered, C: Heterogeneous, D: Dense)
  7. `Left Finding` — Categorical status (`Negative/Benign` vs `Suspicious/Malignant`)
  8. `Right Malignancy (%)` — Probability score for right breast
  9. `Right Breast Density` — ACR density class for right breast
  10. `Right Finding` — Categorical status for right breast
  11. `Single View Duplication` — Flags if a view was duplicated (Left / Right / None)
  12. `Flagged (>5%)` — Soft-red highlighted tripwire (`YES`/`NO`) for elevated malignancy risk
  13. `Status / Notes` — Processing notes and excluded corrupt file alerts

### 2. Per-Patient Text Report

Each patient study folder contains `Study_Report_<patient-id>.txt` with clinical findings:

```
=== CLINICAL AI MAMMOGRAPHY STUDY REPORT ===
Patient ID   : 28-n
Folder       : 28-n
Patient Name : MRS DOE J
Age          : 58
Scanned Date : 2026-07-21
--------------------------------------------

--- LEFT BREAST (L) ---
Cancer Probability : 0.4% (Negative/Benign)
Breast Density     : B (Scattered)
--------------------------------------------

--- RIGHT BREAST (R) ---
Cancer Probability : 8.1% (Negative/Benign)
Breast Density     : C (Heterogeneous)
--------------------------------------------
```

### Understanding the Results

- **Cancer Probability**: Percentage score (0-100%) indicating likelihood of
  malignancy. Values >50% are flagged as "Suspicious/Malignant".
- **Breast Density**: ACR classification from A (fatty) to D (dense). Dense
  breasts (C/D) can mask tumors and may require additional imaging.
- **Single-view warning**: When only one view is available, the model
  duplicates it to meet input requirements. Results should be treated as
  lower-confidence.

### Audit Log

An `audit_log.txt` file is created at the project root to track:
- Number of distinct PatientIDs found in each folder
- Warnings if files from different patients appear mixed

---

## Troubleshooting

### Common Issues

#### "No module named 'albumentations'"
```bash
pip install albumentations
```

#### "No module named 'timm'"
```bash
pip install timm
```

#### "'MammoEnsemble' object has no attribute 'all_tied_weights_keys'"
This error occurs with `transformers` version 5.x. Use the pinned version:
```bash
pip install transformers==4.45.2
```

#### "Failed to build wheel for tokenizers"

- **On Linux**: This usually happens when your environment is using a newer Python version (e.g. 3.12+ or 3.14) that lacks pre-built wheels on PyPI, and a C linker (`cc`) is not installed. To fix, recreate the environment using **Python 3.10** with `uv`:
  ```bash
  rm -rf mammo
  uv venv --python 3.10 mammo
  source mammo/bin/activate
  uv pip install -r requirements.txt
  ```
- **On Windows**: Install Microsoft Visual C++ Build Tools:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/
  
  Or use Conda instead of pip:
  ```bash
  conda create -n mammo python=3.10 -y
  conda activate mammo
  pip install -r requirements.txt
  ```

#### CUDA out of memory
The script will automatically fall back to CPU if GPU memory is insufficient.
To force CPU mode, set:
```bash
export CUDA_VISIBLE_DEVICES=""
python batch-dicom-processor-v4.py --device cpu
```

---

## How it works

1. **Discover** — v4 scans the input path for DICOM files (flat folder, nested
   tree, or pre-organized patient subfolders). Files are grouped into clinical
   studies by `StudyInstanceUID` (fallback: `PatientID` + `StudyDate`).
2. **Read** — each DICOM is parsed with the model's built-in loader
   (windowing / VOI LUT / normalization handled natively).
3. **Normalize** — pixel data is min-max normalized to 8-bit grayscale.
4. **Crop** — `crop_breast_roi` isolates the largest connected tissue region,
   removing background dead space and label burns.
5. **Bucket** — images are sorted into CC/MLO slots per breast from DICOM
   metadata; unknown views are excluded from inference.
6. **Complete** — missing views are duplicated (and flagged) so every breast
   has the two-view pair the ensemble expects.
7. **Infer** — each breast's two-view pair is passed through the model to get
   cancer probability and density class.
8. **Report** — per-patient text/PNG outputs, consolidated Excel summary, and
   flagged-patient PDF reports are written to `output/`.

---

## Credits & Acknowledgments

This project would not exist without the open work of **Dr. Ian Pan**,
radiologist and deep-learning practitioner, who published the `mammoscreen`
model openly on the Hugging Face Hub:

| | |
| --- | --- |
| **Personal website** | [https://ianpan.me/](https://ianpan.me/) |
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

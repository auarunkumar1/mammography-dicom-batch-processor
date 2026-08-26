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

- **Python 3.10** (recommended — other versions may have compatibility issues)
- **~500 MB free disk space** for dependencies and model weights
- **Operating system**: Windows, macOS, or Linux

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `transformers` (pinned to 4.45.x) | Hugging Face model loading |
| `timm` | Pre-trained image model architectures |
| `albumentations` | Image augmentation library |
| `pydicom` | DICOM file parsing |
| `opencv-python` | Image processing and breast ROI cropping |
| `numpy` | Numerical operations |
| `Pillow` | PNG image saving |

---

## Installation (Step-by-Step)

### Option A: Using Conda (Recommended)

Conda is the easiest way to manage dependencies, especially on Windows where
some packages (like `tokenizers`) require special build tools.

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
pip install torch torchvision transformers==4.45.2 timm albumentations pydicom opencv-python numpy Pillow
```

> **Note**: `transformers` is pinned to version 4.45.2 for compatibility with
> the mammoscreen model. Newer versions (5.x) may cause errors.

#### Step 5: Verify installation

```bash
python -c "import torch; import transformers; import timm; import albumentations; print('All dependencies installed successfully!')"
```

You should see: `All dependencies installed successfully!`

---

### Option B: Using pip (Alternative)

If you prefer not to use conda, you can use pip directly:

```bash
# Create a virtual environment (optional but recommended)
python -m venv mammo-env

# Activate it
# Windows:
mammo-env\Scripts\activate
# macOS/Linux:
source mammo-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Windows users**: If you encounter errors building `tokenizers` from source,
> make sure you have **Microsoft Visual C++ Build Tools** installed:
> https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

### Option C: Using Docker (Advanced)

For reproducible environments across different machines:

```bash
docker run -it --rm -v $(pwd):/app -w /app python:3.10-slim bash
pip install -r requirements.txt
python batch-dicom-processor-v3.py
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

> **Tip**: Folder names should not contain any patient-identifying information.
> Use anonymized identifiers like `patient-001`, `study-A`, etc.

### 2. Run the processor

```bash
# Make sure the mammo environment is active
conda activate mammo

# Run the processor
python batch-dicom-processor-v3.py
```

On the **first run**, the script downloads the model weights (~244 MB) from the
Hugging Face Hub and caches them locally; all subsequent runs work **fully
offline**. For strict offline deployment (e.g. a mobile van), add
`local_files_only=True` to the `AutoModel.from_pretrained(...)` call.

The script automatically uses a CUDA GPU when available and falls back to CPU.

### 3. Check results

Results are saved in the `output/` folder, organized by patient ID:

```
output/
└── 28-n/
    ├── Study_Report_28-n.txt    # AI analysis report
    ├── study_01_L_CC.png        # Left breast, CC view
    └── study_02_R_MLO.png       # Right breast, MLO view
```

---

## Output

### Report Format

Each patient folder contains a text report with AI predictions:

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

#### "Failed to build wheel for tokenizers" (Windows)
Install Microsoft Visual C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Or use conda instead of pip:
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
python batch-dicom-processor-v3.py
```

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

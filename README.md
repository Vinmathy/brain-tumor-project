# Brain Tumour MRI Segmentation (3D)

## About
A step-by-step learning project to build a 3D brain tumour segmentation model
using multi-modal MRI scans. The goal is to train a model that labels every
voxel of a brain scan as healthy tissue or a tumour sub-region, and to build a
simple Python viewer to visualise the predicted tumour regions.

## Dataset
**BraTS 2020** (Brain Tumor Segmentation Challenge), from the Kaggle dataset
*"BraTS2020 Dataset (Training + Validation)"* (CC0: Public Domain).

- 369 training patients
- Each patient has 4 MRI scans (NIfTI `.nii`, size 240 × 240 × 155):
  - `flair` – highlights swelling (oedema)
  - `t1` – normal anatomy
  - `t1ce` – T1 with contrast, highlights the active tumour
  - `t2` – fluid appears bright
- 1 segmentation mask (`seg`) with labels:
  - `0` – background / healthy tissue
  - `1` – necrotic tumour core
  - `2` – peritumoural oedema
  - `4` – enhancing tumour
  - (label `3` is not used in BraTS 2020)

> The data is not included in this repository because of its size.
> Download it from Kaggle and place it in a `data/` folder.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install numpy nibabel matplotlib
python explore.py
```

## Progress
- [x] Step 1: Dataset – found and understood the BraTS 2020 data
- [x] Step 2: Python setup – virtual environment and libraries
- [x] Step 3: Exploring scans – compared 4 MRI modalities, overlaid tumour masks, measured class imbalance (tumour = 2.37% of voxels)
- [ ] Step 4: Train model
- [ ] Step 5: Viewer

## Tools
Python 3.12 · NumPy · NiBabel · Matplotlib
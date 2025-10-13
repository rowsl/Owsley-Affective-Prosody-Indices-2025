# Prosody in the Social Feedback Loop in Depression
*Final Clean Pipeline Notebook and Analysis Bundle*

## Overview
This repository contains the reproducible notebook and dependencies for the paper:

> **Prosody and the Social Feedback Loop in Depression**  
> The notebook computes and visualizes two derived indices of prosodic change:
> - **Affective Responsivity Index (ARI):** Within-subject expressivity.
> - **Temporal Responsivity Index (TRI):** Timing adaptability and repair.
>
> Both indices are derived from standardized **COVAREP** and **eGeMAPS** features in the DAIC-WOZ dataset.  
> The dataset did not include column headers, so standardized feature headers were applied for consistency.

The analysis includes robust preprocessing (median absolute deviation), descriptive statistics, visualizations, and PHQ correlation analysis.
The Affective Responsivity Index (ARI) models within-subject expressivity using robust standardized prosodic features.
The final analysis retained **F0** and **Loudness** after dropping **Jitter** and **Shimmer**, which contributed minimally to variance explained.
  
A **Future Directions** section discusses a conceptual **Affective Adaptivity Index (AAI)** for later work.

---

## Contents
| File | Description |
|------|--------------|
| `prosody_paper_final_pipeline.ipynb` | Main Jupyter notebook — clean, runnable pipeline for ARI/TRI analysis. |
| `requirements.txt` | Python dependencies for reproducibility. |
| *(optional)* `final_with_ARI_TRI.csv` | Output dataset with computed indices (generated when you run the notebook). |

---

## Setup Instructions

### 1. Create environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the notebook
Launch JupyterLab or Jupyter Notebook and open:
```bash
jupyter lab prosody_paper_final_pipeline.ipynb
```
Execute cells from top to bottom to reproduce all results.

---

## Expected Outputs
- **Computed indices:** `ARI`, `ARI_F0Loud`, and `TRI`
- **Descriptive statistics:** Summary tables and PHQ correlations
- **Visualization:** Boxplot of ARI_F0Loud across PHQ-8 bands
- **Export:** Clean CSV with derived indices for further analysis

---

## Reproducibility Checklist
- ✅ All required packages are listed in `requirements.txt`  
- ✅ Dataset path variable (`path`) easily adjustable for your environment  
- ✅ Outputs cleared for fresh execution  
- ✅ No redundant or debug cells  
- ✅ Markdown commentary explains each analysis step

---

## Citation
If you reuse or adapt this notebook, please cite the parent paper:

> Owsley, R. (2025). *Prosody and the Social Feedback Loop in Depression.*  
> [Manuscript in preparation]

---

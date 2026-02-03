# scLightGAT : C-DVAE and GAT-Integrated LightGBM for Robust Single-Cell RNA-Seq Cell Type Annotation

## Overview
We propose **scLightGAT**, a Python-native, biologically informed, and computationally scalable framework for cell-type annotation. scLightGAT combines machine learning and deep learning techniques through a three-stage architecture:
1. **C-DVAE**: Contrastive Denoising Variational Autoencoder extracts low-dimensional latent features from highly variable genes (HVGs).
2. **LightGBM**: A gradient-boosted classifier uses the fused latent (Z) and DGE marker (M_DGE) features for an initial cell-type prediction.
3. **GATs**: Graph Attention Networks refine LightGBM’s output by modeling neighborhood interactions on a single-cell graph (SCG).

---

## Installation

```bash
git clone https://github.com/chenh2lab/scLightGAT.git
cd scLightGAT
pip install -r requirements.txt
pip install -e .
```

## Data Availability & Setup

The training and testing datasets for scLightGAT are hosted on Hugging Face.

**Option 1: Automatic Setup (Recommended)**
Run the included helper script to download and place data in the correct directory structure:
```bash
# Downloads data to ../data/scLightGAT_data (or ./data/scLightGAT_data)
./download_hf_data.sh
```

**Option 2: Manual Download**
Download manually from [Hugging Face Datasets](https://huggingface.co/datasets/Alfiechuang/scLightGAT) and place the folders in `data/scLightGAT_data/`.

**Option 3: Python Access**
```python
from huggingface_hub import hf_hub_download
import scanpy as sc

path = hf_hub_download(repo_id="Alfiechuang/scLightGAT", filename="Integrated_training/train.h5ad", repo_type="dataset")
adata = sc.read_h5ad(path)
```

---

## Quick Start (Shell Script)

We recommend using the provided wrapper script `run_sclight.gat.sh` for all operations. It handles environment setup (if configured) and simplifies argument passing.

### 1. Standard Training
Run on a specific dataset using default settings (DVAE=5 epochs, GAT=300 epochs):

```bash
./run_sclight.gat.sh GSE115978
```

Run on **all** available datasets:
```bash
./run_sclight.gat.sh
```

### 2. Hyperparameter Optimization (Optuna)
To automatically tune parameters (LightGBM, DVAE, GAT) for maximum accuracy on your training data:

```bash
# Recommended: 20-30 trials
./run_sclight.gat.sh --optimize --optuna-trials 20 sapiens_full
```
This runs the **scLightGAT Optimization Pipeline**, displaying progress bars for each model and saving the best parameters to `dvae_optimization_results.json` and `gat_optimization_results.json`.

### 3. Hierarchical Classification (Subtypes)
For datasets requiring fine-grained annotation (CD4+T, CD8+T, B cells, Plasma, DC subtypes), use the `--hierarchical` flag:

```bash
./run_sclight.gat.sh --hierarchical GSE115978
```
*Requires `Celltype_subtraining` column in training data.*

### 4. CAF Mode (Cancer-Associated Fibroblasts)
Run training/testing on specific CAF datasets:

```bash
./run_sclight.gat.sh --caf
```

### 5. Inference-Only Mode
Skip training and use pre-trained models saved in `saved_models/`:

```bash
./run_sclight.gat.sh --inference-only GSE123139
```

---

## Advanced Usage

### Custom Ground Truth & Batch Keys
The pipeline automatically detects ground truth columns (e.g., `Manual_celltype`, `final_celltype`). You can also specify them manually:

```bash
./run_sclight.gat.sh --gt-key "My_Labels" --batch-key "sample_id" GSE153935
```

### Key Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--dvae-epochs` | Epochs for C-DVAE | 5 |
| `--gat-epochs` | Epochs for GAT | 300 |
| `--optimize` | Enable Optuna optimization | False |
| `--optuna-trials`| Number of optimization trials | 10 |
| `--hierarchical` | Enable subtype prediction | False |
| `--caf` | Use CAF-specific data | False |
| `--inference-only`| Use pre-trained models | False |

---

## Experimental Results

Results are saved to `sclightgat_exp_results/<DATASET_NAME>/<TIMESTAMP>/`, containing:
- **`adata_with_predictions.h5ad`**: Final AnnData with `scLightGAT_pred`.
- **`accuracy_report.txt`**: Detailed accuracy metrics.
- **`umap_comparison.png`**: Side-by-side UMAP of Ground Truth vs scLightGAT.
- **`scLightGAT_run.log`**: Full execution logs (including debug info).
- **Optimization Results**: JSON files with best parameters (if optimization was run).

---

## Data Structure

### Training Data (`train.h5ad`)
- `.X`: Log-transformed expression matrix.
- `.obs['Celltype_training']`: Broad cell type labels.
- `.obs['Celltype_subtraining']`: Subtype labels (for Hierarchical mode).

### Test Data (`<dataset>.h5ad`)
- `.X`: Log-transformed expression matrix.
- `.obsm['X_umap']`: UMAP coordinates (automatically computed if missing).

---

## Authors
**Tsung-Hsien Chuang**, **Cheng-Yu Li**, **Liang-Chuan Lai**, **Tzu-Pin Lu**, **Mong-Hsun Tsai**, **Eric Y. Chuang***, and **Hsiang-Han Chen***

*Correspondence to: Hsiang-Han Chen (chenh2@ntnu.edu.tw) and Eric Y. Chuang (chuangey@ntu.edu.tw).*

---
© 2025 scLightGAT Team. All rights reserved.

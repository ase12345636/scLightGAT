#!/bin/bash
# ============================================================================
# scLightGAT Accuracy Evaluation Script
# ============================================================================
#
# This script evaluates the prediction accuracy of scLightGAT on independent
# test datasets by comparing 'Celltype_training' (ground truth) with 
# 'scLightGAT_pred' (predictions).
#
# Usage:
#   chmod +x test_exp.sh
#   ./test_exp.sh [RESULT_DIR]
#
# Arguments:
#   RESULT_DIR  Optional. Directory containing result h5ad files.
#               Default: /Group16T/common/lcy/dslab_lcy/GitRepo/scLightGAT/sclightgat_exp_results
#
# ============================================================================

set -e

# ============================================================================
# Conda Environment Setup
# ============================================================================
CONDA_ENV="scgat"
CONDA_PATH="/Group16T/common/lcy/miniconda3"

# Initialize conda
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Using Python: $(which python3)"

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT="/Group16T/common/lcy/dslab_lcy/GitRepo/scLightGAT"
DEFAULT_RESULT_DIR="${PROJECT_ROOT}/sclightgat_exp_results"
SCLIGHTGAT_DIR="${PROJECT_ROOT}/scLightGAT.main"

# Ground truth and prediction column names (check both for compatibility)
GROUND_TRUTH_COL="Ground Truth"
GROUND_TRUTH_COL_LEGACY="Celltype_training"
PREDICTION_COL="scLightGAT_pred"

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
}

evaluate_accuracy() {
    local result_dir="$1"
    
    cd "${SCLIGHTGAT_DIR}"
    
    python3 << PYTHON_SCRIPT
import os
import sys
import scanpy as sc
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

result_dir = "${result_dir}"
sclightgat_dir = "${SCLIGHTGAT_DIR}"
prediction_col = "${PREDICTION_COL}"

# Add scLightGAT to path
sys.path.insert(0, sclightgat_dir)
from scLightGAT.label_utils import calculate_accuracy, generate_accuracy_report

print(f"\nEvaluating results in: {result_dir}")
print(f"Using custom accuracy calculation with myeloid grouping and fine-grained tolerance")
print()

# Find all h5ad files (including subdirectories)
h5ad_files = []
for root, dirs, files in os.walk(result_dir):
    for f in files:
        if f.endswith('.h5ad'):
            h5ad_files.append(os.path.join(root, f))

h5ad_files = sorted(h5ad_files)

if not h5ad_files:
    print("No h5ad files found in the result directory.")
    sys.exit(1)

# Store results for summary
results = []

for filepath in h5ad_files:
    filename = os.path.basename(filepath)
    dataset_name = os.path.basename(os.path.dirname(filepath))
    
    print("=" * 80)
    print(f"Dataset: {dataset_name}/{filename}")
    print("=" * 80)
    
    try:
        adata = sc.read_h5ad(filepath)
        print(f"Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
        
        # Determine ground truth column
        if 'Ground Truth' in adata.obs.columns:
            ground_truth_col = 'Ground Truth'
        elif 'Celltype_training' in adata.obs.columns:
            ground_truth_col = 'Celltype_training'
        else:
            print(f"Warning: No ground truth column found. Skipping.")
            print()
            continue
            
        if prediction_col not in adata.obs.columns:
            print(f"Warning: '{prediction_col}' column not found. Skipping.")
            print()
            continue
        
        # Get ground truth and predictions
        y_true = adata.obs[ground_truth_col]
        y_pred = adata.obs[prediction_col]
        
        # Remove any NaN values
        mask = pd.notna(y_true) & pd.notna(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # Calculate custom accuracy with rules
        accuracy, details = calculate_accuracy(
            y_true_clean,
            y_pred_clean,
            myeloid_group=True,
            fine_grained_tolerance=True,
            epithelial_tolerance=True,
            return_details=True
        )
        
        print(f"\nCustom Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Cells evaluated: {len(y_true_clean)}")
        print(f"   Match breakdown:")
        for reason, count in sorted(details['reason_breakdown'].items()):
            pct = count / len(y_true_clean) * 100
            print(f"      {reason}: {count} ({pct:.1f}%)")
        
        # Per-class report
        print("\nClassification Report:")
        print("-" * 60)
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        
        # Store result
        results.append({
            'Dataset': dataset_name,
            'Cells': len(y_true),
            'Accuracy': f"{accuracy:.4f}",
            'Accuracy_pct': f"{accuracy*100:.2f}%"
        })
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
    
    print()

# Print summary table
if results:
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
else:
    print("No valid results to report.")

PYTHON_SCRIPT
}

# ============================================================================
# Main
# ============================================================================

main() {
    print_header "scLightGAT Accuracy Evaluation"
    
    # Use provided directory or default
    result_dir="${1:-${DEFAULT_RESULT_DIR}}"
    
    echo "Configuration:"
    echo "  Result directory:    ${result_dir}"
    echo "  Ground truth column: ${GROUND_TRUTH_COL}"
    echo "  Prediction column:   ${PREDICTION_COL}"
    
    # Check if result directory exists
    if [ ! -d "${result_dir}" ]; then
        echo "Error: Result directory not found: ${result_dir}"
        exit 1
    fi
    
    # Run evaluation
    evaluate_accuracy "${result_dir}"
}

# Run main function
main "$@"

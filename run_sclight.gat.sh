#!/bin/bash
# ============================================================================
# scLightGAT Quick Usage Guide and Training Script
# ============================================================================
# 
# This script demonstrates how to use the scLightGAT pipeline for cell type
# annotation. Based on example_usage.ipynb from the project.
#
# Usage:
#   chmod +x run_sclight.gat.sh
#   ./run_sclight.gat.sh [OPTIONS] [DATASET_NAME]
#
# Arguments:
#   DATASET_NAME  Optional. One of: GSE115978, GSE123139, GSE153935, GSE166555, Zhengsorted
#                 If not provided, runs on all datasets.
#
# Options:
#   --dvae-epochs N    Number of DVAE epochs (default: 5)
#   --gat-epochs N     Number of GAT epochs (default: 300)
#   --batch-key KEY    Batch key for Harmony correction (optional)
#   --hierarchical     Enable hierarchical subtype classification
#   --list-batch-keys  Show potential batch keys for each dataset
#   --help             Show help message
#
# Recommended batch keys per dataset (analyzed from data):
#   GSE115978:   samples (32 unique)
#   GSE123139:   Processing (2 unique)
#   GSE153935:   sample_id (18 unique)
#   GSE166555:   case_id (12 unique)
#   Zhengsorted: majority_voting (6 unique)
#
# Hierarchical Mode:
#   When --hierarchical is enabled, after broad cell type prediction,
#   the pipeline predicts subtypes for: CD4+T, CD8+T, B cells, Plasma, DC.
#   Requires training data with 'Celltype_subtraining' column (31 subtypes).
#
# ============================================================================

set -e

# ============================================================================
# Conda Environment Setup
# ============================================================================
CONDA_ENV="scLightGAT"
CONDA_PATH="/Group16T/common/lcy/miniconda3"

# Initialize conda
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Using Python: $(which python3)"

# ============================================================================
# Configuration
# ============================================================================

# Base paths
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCLIGHTGAT_DIR="${PROJECT_ROOT}/scLightGAT"
DATA_DIR="${PROJECT_ROOT}/scLightGAT_data"

# Training data
TRAIN_DATA="${DATA_DIR}/Integrated_training/train.h5ad"

# CAF-specific training/test data
CAF_DATA_DIR="${DATA_DIR}/caf.data"
CAF_TRAIN_DATA="${CAF_DATA_DIR}/caf_train.h5ad"
CAF_TEST_DATA="${CAF_DATA_DIR}/caf_test.h5ad"

# Independent test datasets
TEST_DATA_DIR="${DATA_DIR}/Independent_testing"

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/sclightgat_exp_results"
MODEL_DIR="${SCLIGHTGAT_DIR}/scLightGAT/models"

# Directory for saving trained models (for inference-only mode)
TRAINED_MODELS_DIR="${PROJECT_ROOT}/saved_models"

# Default training parameters
DVAE_EPOCHS=5
GAT_EPOCHS=300
BATCH_KEY=""
USE_DVAE="True"
USE_GAT="True"
HIERARCHICAL="False"
CAF_MODE="False"
INFERENCE_ONLY="False"  # Skip training, use cached models

# ============================================================================
# Test Datasets
# ============================================================================

declare -A DATASETS=(
    ["GSE115978"]="${TEST_DATA_DIR}/GSE115978.h5ad"
    ["GSE123139"]="${TEST_DATA_DIR}/GSE123139.h5ad"
    ["GSE153935"]="${TEST_DATA_DIR}/GSE153935.h5ad"
    ["GSE166555"]="${TEST_DATA_DIR}/GSE166555.h5ad"
    ["Zhengsorted"]="${TEST_DATA_DIR}/Zhengsorted.h5ad"
    ["CAF"]="${CAF_TEST_DATA}"
    ["lung_full"]="${TEST_DATA_DIR}/lung_full.h5ad"
    ["sapiens_full"]="${TEST_DATA_DIR}/sapiens_full.h5ad"
)

# Recommended batch keys per dataset (analyzed from column distributions)
# Criteria: 2-50 unique values, not cell type related
declare -A BATCH_KEYS=(
    ["GSE115978"]="samples"           # 32 unique batches (treatment.group only has 2)
    ["GSE123139"]="Processing"        # 2 unique batches
    ["GSE153935"]="sample_id"         # 18 unique samples
    ["GSE166555"]="case_id"           # 12 unique cases
    ["Zhengsorted"]=""                 # None (using default)
    ["CAF"]=""                         # None for CAF
    ["lung_full"]="status"            # Optimal batch key based on previous analysis
    ["sapiens_full"]="donor"          # Optimal batch key based on previous analysis
)

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
}

print_usage() {
    echo "Usage: $0 [OPTIONS] [DATASET_NAME]"
    echo ""
    echo "Available datasets:"
    for dataset in "${!DATASETS[@]}"; do
        echo "  - $dataset (batch_key: ${BATCH_KEYS[$dataset]:-'none'})"
    done
    echo ""
    echo "Options:"
    echo "  --dvae-epochs N    Number of DVAE epochs (default: 5)"
    echo "  --gat-epochs N     Number of GAT epochs (default: 300)"
    echo "  --batch-key KEY    Batch key for Harmony correction"
    echo "  --list-batch-keys  Show potential batch keys for each dataset"
    echo "  --help             Show this help message"
    echo "  --list             List available datasets"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run on all datasets"
    echo "  $0 GSE115978                    # Run on GSE115978 only"
    echo "  $0 --dvae-epochs 10 GSE115978   # Custom DVAE epochs"
    echo "  $0 --batch-key samples GSE115978  # With batch correction"
}

list_batch_keys() {
    print_header "Potential Batch Keys per Dataset"
    python3 << 'PYTHON_SCRIPT'
import scanpy as sc
import os

test_dir = "/Group16T/common/lcy/dslab_lcy/GitRepo/scLightGAT/data/scLightGAT_data/Independent_testing"

for f in sorted(os.listdir(test_dir)):
    if f.endswith('.h5ad'):
        filepath = os.path.join(test_dir, f)
        adata = sc.read_h5ad(filepath)
        print(f"\n{f}:")
        for col in adata.obs.columns:
            n_unique = adata.obs[col].nunique()
            if 2 <= n_unique <= 50:
                print(f"    {col}: {n_unique} unique values")
PYTHON_SCRIPT
}

run_training() {
    local dataset_name="$1"
    local test_path="$2"
    
    # Determine output path based on hierarchical mode
    local output_path="${OUTPUT_DIR}/${dataset_name}"
    if [[ "${HIERARCHICAL}" == "True" ]]; then
        output_path="${OUTPUT_DIR}/${dataset_name}/hierarchical"
    fi
    
    print_header "Training on ${dataset_name}"
    echo "Test data: ${test_path}"
    echo "Output: ${output_path}"
    echo "DVAE epochs: ${DVAE_EPOCHS}"
    echo "GAT epochs: ${GAT_EPOCHS}"
    echo "Hierarchical: ${HIERARCHICAL}"
    echo "Batch key: ${BATCH_KEY:-'None (no batch correction)'}"
    echo ""
    
    mkdir -p "${output_path}"
    mkdir -p "${TRAINED_MODELS_DIR}"
    
    # Change to output directory so all visualizations are saved there
    cd "${output_path}"
    
    # Build batch_key argument for Python
    if [ -n "${BATCH_KEY}" ]; then
        BATCH_KEY_ARG="batch_key='${BATCH_KEY}'"
    else
        BATCH_KEY_ARG="batch_key=None"
    fi
    
    # Add PYTHONPATH so scLightGAT module can be found
    # We add PROJECT_ROOT because scLightGAT package is inside PROJECT_ROOT
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
    
    python3 << PYTHON_SCRIPT
import os
import sys
import scanpy as sc
import matplotlib.pyplot as plt

# Change to scLightGAT directory for imports
sys.path.insert(0, '${PROJECT_ROOT}')
from scLightGAT.pipeline import train_pipeline

# Run training pipeline (outputs will be saved to current directory = output_path)
adata_result = train_pipeline(
    train_path='${TRAIN_DATA}',
    test_path='${test_path}',
    output_path='${output_path}',
    model_dir='${MODEL_DIR}',
    train_dvae=${USE_DVAE},
    use_gat=${USE_GAT},
    dvae_epochs=${DVAE_EPOCHS},
    gat_epochs=${GAT_EPOCHS},
    hierarchical=${HIERARCHICAL},
    ${BATCH_KEY_ARG}
)

# Load result and generate comparison UMAP
result_path = os.path.join('${output_path}', 'adata_with_predictions.h5ad')
adata = sc.read_h5ad(result_path)

# Set figure directory to output path
sc.settings.figdir = '${output_path}'

# Import label utilities
sys.path.insert(0, '${SCLIGHTGAT_DIR}')
from scLightGAT.label_utils import (
    calculate_accuracy, 
    generate_accuracy_report,
    standardize_labels_series,
    get_aligned_color_palette
)

# Determine Ground Truth column
gt_col = 'Ground Truth' if 'Ground Truth' in adata.obs.columns else 'Celltype_training'

# Create comparison UMAP with aligned colors
if gt_col in adata.obs and 'scLightGAT_pred' in adata.obs:
    # Get aligned color palettes
    gt_palette, pred_palette = get_aligned_color_palette(
        adata.obs[gt_col],
        adata.obs['scLightGAT_pred']
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Ground truth - left panel
    ax1 = axes[0]
    sc.pl.umap(adata, color=gt_col, ax=ax1, show=False, 
               title='Ground Truth', frameon=False,
               legend_loc='right margin', legend_fontsize=10, size=50,
               palette=gt_palette)
    
    # scLightGAT prediction - right panel (with aligned colors)
    ax2 = axes[1]
    sc.pl.umap(adata, color='scLightGAT_pred', ax=ax2, show=False,
               title='scLightGAT', frameon=False,
               legend_loc='right margin', legend_fontsize=10, size=50,
               palette=pred_palette)
    
    plt.tight_layout()
    plt.savefig(os.path.join('${output_path}', 'umap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('UMAP comparison saved to ${output_path}/umap_comparison.png')
    
    # Generate separate scLightGAT-only UMAP
    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.umap(adata, color='scLightGAT_pred', ax=ax, show=False,
               title='scLightGAT', frameon=False,
               legend_loc='right margin', legend_fontsize=10, size=50,
               palette=pred_palette)
    plt.tight_layout()
    plt.savefig(os.path.join('${output_path}', 'umap_scLightGAT.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('scLightGAT UMAP saved to ${output_path}/umap_scLightGAT.png')
    
    # Calculate accuracy with custom rules
    # Rule 1.1: Myeloid group (Macrophages, Monocytes, DC) are equivalent
    # Rule 1.3: Fine-grained predictions are correct for broad GT
    # Rule 1.4: Epithelial cells predictions are never wrong
    accuracy, details = calculate_accuracy(
        adata.obs[gt_col],
        adata.obs['scLightGAT_pred'],
        myeloid_group=True,
        fine_grained_tolerance=True,
        epithelial_tolerance=True,
        return_details=True
    )
    
    # Generate and save accuracy report
    report = generate_accuracy_report(
        adata.obs[gt_col],
        adata.obs['scLightGAT_pred'],
        dataset_name='${dataset_name}'
    )
    
    print()
    print(report)
    
    # Save report to file
    with open(os.path.join('${output_path}', 'accuracy_report.txt'), 'w') as f:
        f.write(report)
    
    print(f'Accuracy report saved to ${output_path}/accuracy_report.txt')
PYTHON_SCRIPT
    
    echo ""
    echo "Completed training on ${dataset_name}"
    echo "   Results saved to: ${output_path}"
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Parse command line options
    DATASET_NAME=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dvae-epochs)
                DVAE_EPOCHS="$2"
                shift 2
                ;;
            --gat-epochs)
                GAT_EPOCHS="$2"
                shift 2
                ;;
            --batch-key)
                BATCH_KEY="$2"
                shift 2
                ;;
            --caf)
                CAF_MODE="True"
                shift
                ;;
            --hierarchical)
                HIERARCHICAL="True"
                shift
                ;;
            --inference-only)
                INFERENCE_ONLY="True"
                shift
                ;;
            --list-batch-keys)
                list_batch_keys
                exit 0
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            --list)
                echo "Available datasets:"
                for dataset in "${!DATASETS[@]}"; do
                    echo "  - $dataset: ${DATASETS[$dataset]}"
                done
                exit 0
                ;;
            -*)
                echo "Error: Unknown option: $1"
                print_usage
                exit 1
                ;;
            *)
                DATASET_NAME="$1"
                shift
                ;;
        esac
    done

    print_header "scLightGAT Training Pipeline"
    
    echo "Configuration:"
    if [[ "${CAF_MODE}" == "True" ]]; then
        TRAIN_DATA="${CAF_TRAIN_DATA}"
        echo "  Mode:          CAF (Cancer-Associated Fibroblasts)"
        echo "  Train data:    ${CAF_TRAIN_DATA}"
    else
        echo "  Mode:          Standard"
        echo "  Train data:    ${TRAIN_DATA}"
    fi
    echo "  DVAE epochs:   ${DVAE_EPOCHS}"
    echo "  GAT epochs:    ${GAT_EPOCHS}"
    echo "  Batch key:     ${BATCH_KEY:-'None'}"
    echo "  Use DVAE:      ${USE_DVAE}"
    echo "  Use GAT:       ${USE_GAT}"
    echo "  Hierarchical:  ${HIERARCHICAL}"
    echo "  Inference-only: ${INFERENCE_ONLY}"
    echo ""
    
    # Check if training data exists
    if [ ! -f "${TRAIN_DATA}" ]; then
        echo "Error: Training data not found: ${TRAIN_DATA}"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Run on specific dataset or all datasets
    # Run on specific dataset or interactive CAF mode or all datasets
    if [ -n "${DATASET_NAME}" ]; then
        # Check if it is a pre-defined dataset
        if [ -n "${DATASETS[$DATASET_NAME]}" ]; then
            test_path="${DATASETS[$DATASET_NAME]}"
        else
            # Allow using a direct file path if not a pre-defined name
            if [ -f "${DATASET_NAME}" ]; then
                test_path="${DATASET_NAME}"
                # Extract filename without extension for report naming
                filename=$(basename "${test_path}")
                DATASET_NAME="${filename%.*}"
            else
                echo "Error: Unknown dataset name or file not found: ${DATASET_NAME}"
                print_usage
                exit 1
            fi
        fi
        
        # Auto-detect batch key if not provided
        if [ -z "${BATCH_KEY}" ] && [ -n "${BATCH_KEYS[$DATASET_NAME]}" ]; then
             BATCH_KEY="${BATCH_KEYS[$DATASET_NAME]}"
             echo "Auto-detected batch key for ${DATASET_NAME}: ${BATCH_KEY}"
        fi
        
        run_training "${DATASET_NAME}" "${test_path}"

    elif [[ "${CAF_MODE}" == "True" ]]; then
        # Interactive CAF Mode
        print_header "CAF Interactive Mode"
        echo "Using CAF training data: ${CAF_TRAIN_DATA}"
        echo "This data will be used to annotate your independent test set."
        echo ""
        echo "Please provide the absolute path to your independent test dataset (.h5ad):"
        read -p "> " custom_test_path
        
        # Remove quotes if user added them
        custom_test_path="${custom_test_path%\"}"
        custom_test_path="${custom_test_path#\"}"
        
        if [ ! -f "${custom_test_path}" ]; then
            echo "Error: File not found: ${custom_test_path}"
            exit 1
        fi
        
        # Derive a dataset name from the filename
        filename=$(basename "${custom_test_path}")
        custom_dataset_name="${filename%.*}"
        
        echo "detected dataset name: ${custom_dataset_name}"
        run_training "${custom_dataset_name}" "${custom_test_path}"
        
    elif [[ "${OPTIMIZE}" == "True" ]]; then
        # Optimization Mode
        if [ -z "${DATASET_NAME}" ]; then
             echo "Error: Optimization requires a dataset name (e.g. ./run_sclight.gat.sh --optimize sapiens_full)"
             exit 1
        fi
        
        # We need to find the dataset path first (re-using logic from above but simpler)
        if [ -n "${DATASETS[$DATASET_NAME]}" ]; then
            test_path="${DATASETS[$DATASET_NAME]}"
        else
            echo "Error: Unknown dataset for optimization: ${DATASET_NAME}"
            exit 1
        fi
        
        print_header "Running Optuna Optimization on ${DATASET_NAME}"
        echo "Trials: ${OPTUNA_TRIALS}"
        
        export PYTHONPATH="${SCLIGHTGAT_DIR}:${PYTHONPATH}"
        
        # Call the optimization script (restored from backup logic)
        python3 -m scLightGAT.training.dvae_optuna \
            --data_path "${test_path}" \
            --n_trials "${OPTUNA_TRIALS}" \
            --save_dir "${OUTPUT_DIR}/${DATASET_NAME}"
            
    else
        echo "Running on all pre-defined datasets..."
        for dataset_name in "${!DATASETS[@]}"; do
            test_path="${DATASETS[$dataset_name]}"
            if [ -f "${test_path}" ]; then
                run_training "${dataset_name}" "${test_path}"
            else
                echo "Warning: Skipping ${dataset_name}, file not found: ${test_path}"
            fi
        done
    fi
    
    print_header "All Training Complete"
    echo "Results are saved in: ${OUTPUT_DIR}"
}

# Run main function
main "$@"

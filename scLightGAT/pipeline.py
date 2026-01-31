# scLightGAT Pipeline
# Path: scLightGAT/pipeline.py

import argparse
import os
import scanpy as sc
import torch
from scLightGAT.logger_config import setup_logger
from scLightGAT.training.model_manager import CellTypeAnnotator 
from scLightGAT.visualization.visualization import plot_umap_by_label

logger = setup_logger(__name__)

# Label mapping from test data formats to training data format
LABEL_MAPPING = {
    # Space variations in T cell labels
    'CD4+ T cells': 'CD4+T cells',
    'CD8+ T cells': 'CD8+T cells',
    'CD4 T cells': 'CD4+T cells',
    'CD8 T cells': 'CD8+T cells',
    # Typos and variations
    'Monocyets/Macrophages': 'Macrophages',
    'Monocytes/Macrophages': 'Macrophages',
    'MonocytesMacrophages': 'Macrophages',
    # Generic T cells - map to CD4+T cells as default (most common)
    'T cells': 'CD4+T cells',
    # Other potential variations
    'T/NK cells': 'NK cells',
    'cDC': 'DC',
}

def standardize_celltype_labels(adata, label_col=None, train_labels=None):
    """
    Standardize cell type labels to match training data format.
    
    Args:
        adata: AnnData object to modify
        label_col: Column name containing cell type labels (auto-detected if None)
        train_labels: Optional set of valid training labels for validation
        
    Returns:
        Modified AnnData with standardized labels
    """
    # Auto-detect label column
    if label_col is None:
        if 'Ground Truth' in adata.obs.columns:
            label_col = 'Ground Truth'
        elif 'Celltype_training' in adata.obs.columns:
            label_col = 'Celltype_training'
        else:
            logger.warning("No Ground Truth or Celltype_training column found")
            return adata
    
    # Convert to string to avoid Categorical issues
    adata.obs[label_col] = adata.obs[label_col].astype(str)
    
    # Apply mapping
    mapped_count = 0
    for old_label, new_label in LABEL_MAPPING.items():
        mask = adata.obs[label_col] == old_label
        if mask.any():
            count = mask.sum()
            adata.obs.loc[mask, label_col] = new_label
            logger.info(f"Label mapping: '{old_label}' -> '{new_label}' ({count} cells)")
            mapped_count += count
    
    if mapped_count > 0:
        logger.info(f"Total cells with standardized labels: {mapped_count}")
    else:
        logger.info("No label standardization needed")
    
    return adata

def train_pipeline(train_path: str, test_path: str, output_path: str, model_dir: str, train_dvae: bool, use_gat: bool, dvae_epochs: int, gat_epochs: int, hierarchical: bool = False, batch_key: str = None):
    logger.info("[TRAIN MODE] Starting training pipeline")
    logger.info(f"Train data: {train_path}")
    logger.info(f"Test data: {test_path}")
    logger.info(f"Output dir: {output_path}")
    logger.info(f"Model save dir: {model_dir}")
    logger.info(f"Train DVAE: {train_dvae}, Use GAT: {use_gat}, DVAE Epochs: {dvae_epochs}, GAT Epochs: {gat_epochs}")
    logger.info(f"Hierarchical mode: {hierarchical}")
    logger.info(f"Batch key for Harmony: {batch_key if batch_key else 'None (no batch correction)'}")

    adata_train = sc.read_h5ad(train_path)
    adata_test = sc.read_h5ad(test_path)
    adata_train.raw = adata_train.copy()
    adata_test.raw = adata_test.copy()
    
    # Standardize cell type labels in test data to match training format
    logger.info("Standardizing cell type labels in test data...")
    # Determine which column to use for ground truth
    gt_col_train = 'Ground Truth' if 'Ground Truth' in adata_train.obs.columns else 'Celltype_training'
    gt_col_test = 'Ground Truth' if 'Ground Truth' in adata_test.obs.columns else 'Celltype_training'
    
    train_labels = set(adata_train.obs[gt_col_train].unique())
    adata_test = standardize_celltype_labels(adata_test, gt_col_test, train_labels)
    
    annotator = CellTypeAnnotator(
        use_dvae=train_dvae,
        use_hvg=True,
        hierarchical=hierarchical,
        dvae_params={'epochs': dvae_epochs},
        gat_epochs=gat_epochs
    )

    # Run pipeline with optional batch_key for Harmony batch correction
    adata_result, dvae_losses, gat_losses = annotator.run_pipeline(
        adata_train=adata_train,
        adata_test=adata_test,
        save_visualizations=True
    )

    adata_result.write(os.path.join(output_path, "adata_with_predictions.h5ad"))
    logger.info("[TRAIN MODE] Training pipeline completed")
    
    return adata_result

def predict_pipeline(train_path: str, test_path: str, output_path: str, model_dir: str):
    logger.info("[PREDICT MODE] Running inference pipeline")
    logger.info(f"Test data: {test_path}")
    logger.info(f"Using models from: {model_dir}")

    adata_test = sc.read_h5ad(test_path)
    
    
    annotator = CellTypeAnnotator()

    adata_result = annotator.run_inference(
        adata_test=adata_test,
        model_dir=model_dir
    )

    adata_result.write(os.path.join(output_path, "adata_predicted.h5ad"))
    logger.info("[PREDICT MODE] Prediction pipeline completed")

def main():
    parser = argparse.ArgumentParser(description="scLightGAT: Cell Type Annotation Pipeline")
    parser.add_argument('--train', type=str, required=True, help='Path to training .h5ad file')
    parser.add_argument('--test', type=str, required=True, help='Path to testing .h5ad file')
    parser.add_argument('--output', type=str, required=True, help='Directory to save results')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'visualize'], help='Execution mode')
    parser.add_argument('--model_dir', type=str, default='models/', help='Path to save/load models')
    parser.add_argument('--train_dvae', action='store_true', help='Flag to enable DVAE training (default: False)')
    parser.add_argument('--use_gat', action='store_true', help='Flag to enable GAT refinement after LightGBM')
    parser.add_argument('--dvae_epochs', type=int, default=15, help='Number of epochs for DVAE training')
    parser.add_argument('--gat_epochs', type=int, default=300, help='Number of epochs for GAT training')
    parser.add_argument('--hierarchical', action='store_true', help='Enable hierarchical classification')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    if args.mode == 'train':
        train_pipeline(args.train, args.test, args.output, args.model_dir, 
                       args.train_dvae, args.use_gat, args.dvae_epochs, args.gat_epochs,
                       args.hierarchical)
    elif args.mode == 'predict':
        predict_pipeline(args.train, args.test, args.output, args.model_dir)
    elif args.mode == 'visualize':
        visualize_pipeline(args.test, args.output)
    else:
        raise ValueError("Invalid mode")

if __name__ == "__main__":
    main()
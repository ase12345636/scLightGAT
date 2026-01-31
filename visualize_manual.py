import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from scLightGAT.logger_config import setup_logger, setup_warning_logging
from scLightGAT.visualization.visualization import plot_umap_by_label

# Setup logging
setup_warning_logging()
logger = setup_logger("VisualizeManual")

def visualize_dataset(name, path, label_col='Manual_celltype'):
    logger.info(f"Visualizing {name} from {path}")
    
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return

    try:
        adata = sc.read_h5ad(path)
        logger.info(f"Loaded {name}: {adata.shape}")
        
        if label_col not in adata.obs.columns:
            logger.warning(f"Sort of expected: {label_col} not found in {name}. Available: {adata.obs.columns}")
            return

        # Ensure UMAP exists
        if 'X_umap' not in adata.obsm:
            logger.info("Computing UMAP...")
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            
        # Plot
        save_path = f"{name}_{label_col}_umap.png"
        logger.info(f"Plotting UMAP colored by {label_col} to {save_path}")
        
        # Use existing visualization function if possible or custom logic
        # We can use sc.pl.umap directly or the helper
        
        plt.figure(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color=label_col,
            title=f"{name} - {label_col}",
            frameon=False,
            legend_loc='right margin',
            show=False,
            save=False 
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization to {save_path}")

    except Exception as e:
        logger.error(f"Error visualizing {name}: {e}")

def main():
    datasets = {
        "lung_full": "scLightGAT_data/Independent_testing/lung_full.h5ad",
        "sapiens_full": "scLightGAT_data/Independent_testing/sapiens_full.h5ad"
    }

    for name, path in datasets.items():
        visualize_dataset(name, path, 'Manual_celltype')

if __name__ == "__main__":
    main()

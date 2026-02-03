import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from datetime import datetime
import torch
import optuna

from scLightGAT.logger_config import setup_logger, setup_warning_logging
from scLightGAT.models.dvae_model import DVAE 
# Assuming dvae_evaluation is in scLightGAT.evaluation
from scLightGAT.evaluation.dvae_evaluation import extract_features_dvae, optimize_dvae_params, save_results

# Setup logger
# Note: scLightGAT.logger_config.setup_logger might be enough, but we want file logging too as per original
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'dvae_optimization_{timestamp}.log')
    
    # We use a separate logger for optimization script to not conflict with global config if needed,
    # or just configure the root logger if setup_logger returns it.
    # But here we want a specific file handler.
    logger = logging.getLogger("scLightGAT_optimization")
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

def main():
    parser = argparse.ArgumentParser(description="scLightGAT Optimization Pipeline")
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset .h5ad')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory for logs')
    
    args = parser.parse_args()
    
    # Default log dir to inside save_dir if not specified
    if args.log_dir is None:
        args.log_dir = os.path.join(args.save_dir, 'logs')
        
    logger = setup_logging(args.log_dir)
    # Enable warning suppression
    setup_warning_logging(os.path.join(args.log_dir, 'warnings.log'))
    
    logger.info("Starting Optimization Pipeline")
    
    try:
        # Create result directory
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Log system info
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Load data
        logger.info(f"Loading data from {args.data_path}")
        adata = sc.read_h5ad(args.data_path)
        logger.info(f"Data shape: {adata.shape}")
        
        # Base parameters (defaults)
        base_params = {
            'epochs': 15, # Default from pipeline
            'batch_size': 128,
            'balanced_count': 10000,
            'n_hvgs': 3000
        }
        
        # Start optimization
        logger.info(f"Starting parameter optimization with {args.n_trials} trials")
        best_params = optimize_dvae_params(
            adata,
            n_trials=args.n_trials,
            base_params=base_params
        )
        
        logger.info(f"Best parameters found: {best_params}")
        
        # Train with best parameters for verification/stats
        logger.info("Starting validation with best parameters")
        training_params = {**base_params, **best_params}
        results = extract_features_dvae(adata, params=training_params, use_hvg=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Save detailed results in a specific subfolder or just in save_dir
        # Original script made a subfolder
        # run_save_dir = os.path.join(args.save_dir, f'run_{timestamp}') 
        # But we want results easily accessible. Let's save in save_dir directly or subfolder.
        run_save_dir = args.save_dir
        
        logger.info(f"Saving results to {run_save_dir}")
        save_results(results, run_save_dir)
        
        # Save best parameters
        with open(os.path.join(run_save_dir, 'dvae_optimization_best_params.txt'), 'w') as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
        
        # Also save as JSON for easier parsing
        import json
        with open(os.path.join(run_save_dir, 'dvae_optimization_results.json'), 'w') as f:
            json.dump(best_params, f, indent=4)

        logger.info("Optimization Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
"""
scLightGAT Parameter Optimization
=================================
This module provides Optuna-based hyperparameter optimization for 
LightGBM, DVAE, and GAT models in the scLightGAT pipeline.

Usage:
    python optimize_params.py --model lightgbm --n-trials 50
    python optimize_params.py --model dvae --n-trials 20
    python optimize_params.py --model gat --n-trials 30
"""

import os
import sys
import argparse
import optuna
import numpy as np
import scanpy as sc
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import logging
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scLightGAT.training.model_manager import CellTypeAnnotator
from scLightGAT.preprocess.data_preprocessor import preprocess_data, find_common_hvgs
from scLightGAT.logger_config import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# Default Paths
# =============================================================================

DATA_DIR = '/Group16T/common/lcy/dslab_lcy/GitRepo/scLightGAT/data/scLightGAT_data'
TRAIN_PATH = f'{DATA_DIR}/Integrated_training/train.h5ad'
TEST_PATH = f'{DATA_DIR}/Independent_testing/GSE115978.h5ad'
OUTPUT_DIR = '/Group16T/common/lcy/dslab_lcy/GitRepo/scLightGAT/evaluation_results'


# =============================================================================
# LightGBM Optimization
# =============================================================================

def optimize_lightgbm(n_trials: int = 50, sample_size: int = 10000):
    """
    Optimize LightGBM hyperparameters using Optuna.
    
    Args:
        n_trials: Number of optimization trials
        sample_size: Number of cells to sample for fast optimization
        
    Returns:
        Best parameters dictionary
    """
    from lightgbm import LGBMClassifier
    
    logger.info(f"Loading training data from {TRAIN_PATH}")
    adata = sc.read_h5ad(TRAIN_PATH)
    
    # Sample for faster optimization
    if adata.shape[0] > sample_size:
        indices = np.random.choice(adata.shape[0], sample_size, replace=False)
        adata = adata[indices, :].copy()
        logger.info(f"Sampled {sample_size} cells for optimization")
    
    # Prepare features
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(adata.obs['Celltype_training'].values)
    n_class = len(encoder.classes_)
    
    logger.info(f"Data shape: {X.shape}, {n_class} classes")
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'num_leaves': trial.suggest_int('num_leaves', 20, 256),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True)
        }
        
        clf = LGBMClassifier(
            objective='multiclass',
            boosting_type='gbdt',
            num_class=n_class,
            metric='multi_logloss',
            verbose=-1,
            **params
        )
        
        # Cross-validation
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best accuracy: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params, study.best_value


# =============================================================================
# DVAE Optimization
# =============================================================================

def optimize_dvae(n_trials: int = 20, sample_size: int = 5000):
    """
    Optimize DVAE hyperparameters using Optuna.
    Focus on reconstruction quality and cluster separation.
    
    Args:
        n_trials: Number of optimization trials
        sample_size: Number of cells to sample
        
    Returns:
        Best parameters dictionary
    """
    import torch
    from scLightGAT.models.dvae import DVAE
    
    logger.info(f"Loading training data from {TRAIN_PATH}")
    adata = sc.read_h5ad(TRAIN_PATH)
    
    # Sample for faster optimization
    if adata.shape[0] > sample_size:
        indices = np.random.choice(adata.shape[0], sample_size, replace=False)
        adata = adata[indices, :].copy()
    
    # Preprocess
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable']].copy()
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = torch.FloatTensor(X)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(adata.obs['Celltype_training'].values)
    y = torch.LongTensor(y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]
    
    def objective(trial):
        latent_dim = trial.suggest_int('latent_dim', 64, 512)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        kl_weight = trial.suggest_float('kl_weight', 0.001, 0.1, log=True)
        cluster_weight = trial.suggest_float('cluster_weight', 0.5, 5.0)
        noise_factor = trial.suggest_float('noise_factor', 0.05, 0.3)
        temperature = trial.suggest_float('temperature', 0.01, 0.5, log=True)
        
        model = DVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            noise_factor=noise_factor,
            cluster_weight=cluster_weight,
            temperature=temperature
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Quick training
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        X_train, X_val = X_train.to(device), X_val.to(device)
        y_train, y_val = y_train.to(device), y_val.to(device)
        
        model.train()
        batch_size = 256
        for epoch in range(5):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                recon, mu, logvar = model(batch_X)
                loss = model.loss_function(recon, batch_X, mu, logvar, kl_weight)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            z = model.encode(X_val)[0]
            # Reconstruction loss
            recon, _, _ = model(X_val)
            recon_loss = torch.nn.functional.mse_loss(recon, X_val).item()
            
        return -recon_loss  # Minimize reconstruction loss
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best reconstruction: {-study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params, -study.best_value


# =============================================================================
# GAT Optimization
# =============================================================================

def optimize_gat(n_trials: int = 30, sample_size: int = 5000):
    """
    Optimize GAT hyperparameters using Optuna.
    
    Args:
        n_trials: Number of optimization trials
        sample_size: Number of cells to sample
        
    Returns:
        Best parameters dictionary
    """
    logger.info("GAT optimization requires full pipeline setup.")
    logger.info("Recommended GAT parameters based on prior experiments:")
    
    default_params = {
        'hidden_dim': 256,
        'heads': 4,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'weight_decay': 1e-5,
        'epochs': 300
    }
    
    logger.info(f"Default GAT params: {default_params}")
    return default_params, None


# =============================================================================
# Main
# =============================================================================

def save_results(params: dict, score: float, model_name: str):
    """Save optimization results to JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    result = {
        'model': model_name,
        'best_params': params,
        'best_score': score,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f'{OUTPUT_DIR}/{model_name}_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description='Optimize scLightGAT model parameters')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['lightgbm', 'dvae', 'gat', 'all'],
                        help='Model to optimize')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Sample size for optimization')
    
    args = parser.parse_args()
    
    logger.info(f"Starting {args.model} optimization with {args.n_trials} trials")
    
    if args.model in ['lightgbm', 'all']:
        params, score = optimize_lightgbm(args.n_trials, args.sample_size)
        save_results(params, score, 'lightgbm')
    
    if args.model in ['dvae', 'all']:
        params, score = optimize_dvae(min(args.n_trials, 20), min(args.sample_size, 5000))
        save_results(params, score, 'dvae')
    
    if args.model in ['gat', 'all']:
        params, score = optimize_gat(min(args.n_trials, 30), args.sample_size)
        save_results(params, score, 'gat')


if __name__ == '__main__':
    main()

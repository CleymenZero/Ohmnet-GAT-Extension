import os
import random
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Seed for initial setup
set_seed(42)

# Centralized Config Dictionary
config_dict = {
    # Data Paths
    "data_dir": DATA_DIR,
    "hierarchy_path": DATA_DIR / "tissue.hierarchy",
    "obo_path": DATA_DIR / "BrendaTissue.obo",
    "ppi_path": DATA_DIR / "PPT-Ohmnet_tissues-combined.edgelist",
    
    # Model Checkpoint paths
    "model_save_path": MODEL_DIR / "best_model.pt",
    "attention_save_path": MODEL_DIR / "final_attention_coefficients.pt",
    
    # Model Hyperparameters
    "protein_embedding_dim": 256,
    "num_heads": 8,
    "gat_hidden_channels": 256,
    "gat_output_channels": 256,
    "mlp_hidden_channels": 1024,
    "dropout_rate": 0.3,
    
    # Optimization Hyperparameters
    "learning_rate": 0.0005,
    "num_epochs": 1000,
    "patience": 50,
    "batch_size": 1024,
    "initial_lambda": 200,
    "violation_threshold": 0.01,
    "lambda_multiplier": 1.1,
    "max_lambda": 300.0,
    "seed": 42
}

config = SimpleNamespace(**config_dict)
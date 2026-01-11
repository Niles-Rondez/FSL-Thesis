import os
from pathlib import Path

def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: DirectML may not be fully deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Dataset configuration
DATA_ROOT = "data"  # User can change this
SUBJECT_PATTERN = "H*"  # Glob pattern for subject folders
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Classes (A-Z excluding J, Z + 0-9)
CLASSES = sorted([chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in ['J', 'Z']] + 
                 [str(i) for i in range(10)])

# Model configuration
RESNET_MODELS = ["resnet50", "resnet101"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS_LOSO = 15  # Per fold
NUM_EPOCHS_FINAL = 20  # Final model
SEED = 42

# Paths
LOSO_MODELS_DIR = Path("models/loso_folds")
FINAL_MODELS_DIR = Path("models/final")
LOSO_RESULTS_DIR = Path("results/loso_evaluation")
AGGREGATED_RESULTS_DIR = Path("results/aggregated")
INFERENCE_LOGS_DIR = Path("results/inference_logs")

# Create directories
for path in [LOSO_MODELS_DIR, FINAL_MODELS_DIR, LOSO_RESULTS_DIR, 
             AGGREGATED_RESULTS_DIR, INFERENCE_LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
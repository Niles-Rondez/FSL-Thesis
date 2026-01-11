#!/usr/bin/env python3
"""Train final deployment models on all subjects"""
import sys
sys.path.insert(0, ".")

from config import set_seed
set_seed(config.SEED)

import config
from training.final_trainer import FinalTrainer

def main():
    trainer = FinalTrainer()
    
    print("\n" + "="*80)
    print("TRAINING FINAL DEPLOYMENT MODELS")
    print("="*80)
    print(f"Subjects: {trainer.dataset.subjects}")
    print(f"Models: {config.RESNET_MODELS}")
    print("="*80 + "\n")
    
    for model_name in config.RESNET_MODELS:
        trainer.train_final_model(model_name)
    
    print("\n" + "="*80)
    print("FINAL TRAINING COMPLETE")
    print("="*80)
    print(f"Models saved in: {config.FINAL_MODELS_DIR}")
    print("These models are ready for deployment in app.py and fsl_live_cam.py")

if __name__ == "__main__":
    main()
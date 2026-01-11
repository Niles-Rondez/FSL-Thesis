import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import config
from data_management.dataset_loader import SubjectAwareDataset
from training.model_factory import create_model
from device_helper import get_device

class FinalTrainer:
    """Train final deployment models on all subjects"""
    
    def __init__(self):
        self.dataset = SubjectAwareDataset()
        self.device, self.device_type = get_device()
        
    def train_final_model(self, model_name: str):
        """Train model on all subjects for deployment"""
        print(f"\n{'='*60}")
        print(f"FINAL TRAINING: {model_name} on ALL SUBJECTS")
        print(f"{'='*60}")
        
        # Load all data
        train_data = self.dataset.get_all_subjects_data()
        print(f"Total training samples: {len(train_data)}")
        
        pin_memory = self.device_type == "cuda"
        train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, 
                                  shuffle=True, num_workers=0, pin_memory=pin_memory)
        
        # Model
        model = create_model(model_name, len(config.CLASSES)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        best_train_acc = 0.0
        
        for epoch in range(1, config.NUM_EPOCHS_FINAL + 1):
            model.train()
            losses, y_true, y_pred = [], [], []
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.cpu().numpy())
            
            train_loss = np.mean(losses)
            train_acc = accuracy_score(y_true, y_pred)
            
            print(f"Epoch {epoch}/{config.NUM_EPOCHS_FINAL} | "
                  f"Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                save_path = config.FINAL_MODELS_DIR / f"{model_name}_final.pth"
                torch.save({
                    "model_state": model.state_dict(),
                    "train_acc": train_acc,
                    "epoch": epoch,
                    "classes": config.CLASSES,
                    "num_subjects": len(self.dataset.subjects),
                    "note": "Final deployment model trained on all subjects"
                }, save_path)
                print(f"[SAVED] {save_path}")
        
        print(f"\n[COMPLETE] {model_name} final training. Best acc: {best_train_acc:.4f}")
        return best_train_acc
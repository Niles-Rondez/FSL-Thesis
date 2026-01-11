import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from pathlib import Path

import config
from data_management.dataset_loader import SubjectAwareDataset
from training.model_factory import create_model
from device_helper import get_device

class LOSOEvaluator:
    def __init__(self):
        self.dataset = SubjectAwareDataset()
        self.device, self.device_type = get_device()
        self.num_folds = len(self.dataset.subjects)
        
    def run_fold(self, fold: int, model_name: str):
        """Train and evaluate one LOSO fold"""
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{self.num_folds-1} | Model: {model_name}")
        print(f"{'='*60}")
        
        # Load data
        train_data, test_data, test_subject = self.dataset.get_loso_splits(fold)
        print(f"Test subject: {test_subject}")
        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # DataLoaders
        pin_memory = self.device_type == "cuda"
        train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, 
                                  shuffle=True, num_workers=0, pin_memory=pin_memory)
        test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, 
                                 shuffle=False, num_workers=0, pin_memory=pin_memory)
        
        # Model
        model = create_model(model_name, len(config.CLASSES)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                         patience=3, factor=0.5)
        
        # Training loop
        best_test_acc = 0.0
        for epoch in range(1, config.NUM_EPOCHS_LOSO + 1):
            train_loss, train_acc = self._train_epoch(model, train_loader, 
                                                       criterion, optimizer, epoch)
            test_loss, test_acc = self._test_epoch(model, test_loader, criterion)
            scheduler.step(test_acc)
            
            print(f"Epoch {epoch}/{config.NUM_EPOCHS_LOSO} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                model_path = config.LOSO_MODELS_DIR / f"fold_{fold}_{model_name}.pth"
                torch.save({
                    "model_state": model.state_dict(),
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "fold": fold,
                    "test_subject": test_subject,
                    "classes": config.CLASSES
                }, model_path)
        
        # Final evaluation with best model
        best_model_path = config.LOSO_MODELS_DIR / f"fold_{fold}_{model_name}.pth"
        checkpoint = torch.load(best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state"])
        
        y_true, y_pred = self._get_predictions(model, test_loader)
        
        # Save metrics
        self._save_metrics(fold, model_name, test_subject, y_true, y_pred)
        
        return best_test_acc
    
    def _train_epoch(self, model, loader, criterion, optimizer, epoch):
        model.train()
        losses, y_true, y_pred = [], [], []
        
        for images, labels in tqdm(loader, desc=f"Train Epoch {epoch}"):
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
        
        return np.mean(losses), np.mean(np.array(y_true) == np.array(y_pred))
    
    def _test_epoch(self, model, loader, criterion):
        model.eval()
        losses, y_true, y_pred = [], [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                losses.append(loss.item())
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.cpu().numpy())
        
        return np.mean(losses), np.mean(np.array(y_true) == np.array(y_pred))
    
    def _get_predictions(self, model, loader):
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())
        
        return np.array(y_true), np.array(y_pred)
    
    def _save_metrics(self, fold, model_name, test_subject, y_true, y_pred):
        """Save per-fold classification report and confusion matrix"""
        # Classification report CSV
        report = classification_report(y_true, y_pred, target_names=config.CLASSES, 
                                      output_dict=True, zero_division=0)
        
        csv_path = config.LOSO_RESULTS_DIR / f"fold_{fold}_{model_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "test_subject", "class", "precision", 
                            "recall", "f1-score", "support"])
            for cls in config.CLASSES:
                m = report[cls]
                writer.writerow([fold, test_subject, cls, m["precision"], 
                               m["recall"], m["f1-score"], m["support"]])
            writer.writerow([fold, test_subject, "accuracy", report["accuracy"], 
                           "", "", sum([report[c]["support"] for c in config.CLASSES])])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=config.CLASSES, 
                   yticklabels=config.CLASSES)
        plt.title(f"Fold {fold} - {model_name} - Test Subject: {test_subject}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = config.LOSO_RESULTS_DIR / f"fold_{fold}_{model_name}_cm.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        
        print(f"Saved metrics: {csv_path}")
        print(f"Saved confusion matrix: {cm_path}")
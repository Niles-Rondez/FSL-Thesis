import csv
import numpy as np
import pandas as pd
from pathlib import Path
import config

class MetricsAggregator:
    """Aggregate LOSO fold results across all folds"""
    
    def aggregate_loso_results(self, model_name: str):
        """Compute mean ± std metrics across folds"""
        print(f"\nAggregating LOSO results for {model_name}...")
        
        fold_files = sorted(config.LOSO_RESULTS_DIR.glob(f"fold_*_{model_name}.csv"))
        if not fold_files:
            raise FileNotFoundError(f"No LOSO results found for {model_name}")
        
        # Per-class aggregation
        class_metrics = {cls: {"precision": [], "recall": [], "f1-score": []} 
                        for cls in config.CLASSES}
        overall_accs = []
        
        for fold_file in fold_files:
            df = pd.read_csv(fold_file)
            
            # Extract overall accuracy
            acc_row = df[df["class"] == "accuracy"]
            if not acc_row.empty:
                overall_accs.append(acc_row.iloc[0]["precision"])  # Stored in precision col
            
            # Extract per-class metrics
            for cls in config.CLASSES:
                cls_row = df[df["class"] == cls]
                if not cls_row.empty:
                    class_metrics[cls]["precision"].append(cls_row.iloc[0]["precision"])
                    class_metrics[cls]["recall"].append(cls_row.iloc[0]["recall"])
                    class_metrics[cls]["f1-score"].append(cls_row.iloc[0]["f1-score"])
        
        # Save aggregated CSV
        out_path = config.AGGREGATED_RESULTS_DIR / f"{model_name}_loso_summary.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "precision_mean", "precision_std", 
                           "recall_mean", "recall_std", "f1_mean", "f1_std"])
            
            for cls in config.CLASSES:
                m = class_metrics[cls]
                writer.writerow([
                    cls,
                    np.mean(m["precision"]), np.std(m["precision"]),
                    np.mean(m["recall"]), np.std(m["recall"]),
                    np.mean(m["f1-score"]), np.std(m["f1-score"])
                ])
            
            # Overall accuracy
            writer.writerow([
                "OVERALL_ACCURACY",
                np.mean(overall_accs), np.std(overall_accs),
                "", "", "", ""
            ])
        
        print(f"[SAVED] {out_path}")
        print(f"Overall Accuracy: {np.mean(overall_accs):.4f} ± {np.std(overall_accs):.4f}")
        
        return out_path
import pandas as pd
from pathlib import Path
import config

class ResultsViewer:
    """Command-line tool to view and compare LOSO results"""
    
    def compare_models(self):
        """Display side-by-side comparison of ResNet-50 vs ResNet-101"""
        print("\n" + "="*80)
        print("LOSO EVALUATION COMPARISON: ResNet-50 vs ResNet-101")
        print("="*80 + "\n")
        
        for model_name in config.RESNET_MODELS:
            summary_file = config.AGGREGATED_RESULTS_DIR / f"{model_name}_loso_summary.csv"
            if not summary_file.exists():
                print(f"[WARNING] {summary_file} not found. Run aggregation first.")
                continue
            
            df = pd.read_csv(summary_file)
            overall_row = df[df["class"] == "OVERALL_ACCURACY"]
            
            if not overall_row.empty:
                mean_acc = overall_row.iloc[0]["precision_mean"]
                std_acc = overall_row.iloc[0]["precision_std"]
                print(f"{model_name.upper():15s} | Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        print("\nDetailed per-class metrics saved in:")
        print(f"  {config.AGGREGATED_RESULTS_DIR}")
        
    def show_per_class_metrics(self, model_name: str):
        """Display per-class F1 scores"""
        summary_file = config.AGGREGATED_RESULTS_DIR / f"{model_name}_loso_summary.csv"
        if not summary_file.exists():
            print(f"[ERROR] {summary_file} not found.")
            return
        
        df = pd.read_csv(summary_file)
        df = df[df["class"] != "OVERALL_ACCURACY"]
        
        print(f"\nPer-Class F1 Scores for {model_name}:")
        print("-" * 60)
        print(f"{'Class':<10s} {'F1 Mean':<15s} {'F1 Std':<15s}")
        print("-" * 60)
        
        for _, row in df.iterrows():
            print(f"{row['class']:<10s} {row['f1_mean']:<15.4f} {row['f1_std']:<15.4f}")
        
        print("-" * 60)

if __name__ == "__main__":
    viewer = ResultsViewer()
    viewer.compare_models()
    
    print("\nShow detailed metrics? [resnet50/resnet101/skip]: ", end="")
    choice = input().strip().lower()
    if choice in config.RESNET_MODELS:
        viewer.show_per_class_metrics(choice)
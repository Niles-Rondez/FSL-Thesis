#!/usr/bin/env python3
"""Execute full LOSO evaluation pipeline"""
import sys
sys.path.insert(0, ".")

from config import set_seed
set_seed(config.SEED)

import config
from training.loso_evaluator import LOSOEvaluator
from evaluation.metrics_aggregator import MetricsAggregator

def main():
    evaluator = LOSOEvaluator()
    aggregator = MetricsAggregator()
    
    print("\n" + "="*80)
    print("STARTING LOSO CROSS-VALIDATION")
    print("="*80)
    print(f"Number of folds: {evaluator.num_folds}")
    print(f"Subjects: {evaluator.dataset.subjects}")
    print(f"Models: {config.RESNET_MODELS}")
    print("="*80 + "\n")
    
    for model_name in config.RESNET_MODELS:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*80}\n")
        
        fold_accuracies = []
        for fold in range(evaluator.num_folds):
            acc = evaluator.run_fold(fold, model_name)
            fold_accuracies.append(acc)
        
        print(f"\n{model_name} LOSO Results:")
        print(f"  Fold accuracies: {fold_accuracies}")
        print(f"  Mean: {sum(fold_accuracies)/len(fold_accuracies):.4f}")
        
        # Aggregate results
        aggregator.aggregate_loso_results(model_name)
    
    print("\n" + "="*80)
    print("LOSO EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved in: {config.LOSO_RESULTS_DIR}")
    print(f"Aggregated metrics in: {config.AGGREGATED_RESULTS_DIR}")

if __name__ == "__main__":
    main()
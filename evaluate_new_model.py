#!/usr/bin/env python3
"""
Quick script to evaluate any new model
Usage: python evaluate_new_model.py --model "model_name_here"
"""

import argparse
import json
import time
from fast_evaluator import FastModelEvaluator

def main():
    parser = argparse.ArgumentParser(description='Evaluate a new model on Bangla dataset')
    parser.add_argument('--model', type=str, required=True, help='Model name or path (e.g., meta-llama/Llama-3.1-8B)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to evaluate (default: 1000)')
    parser.add_argument('--dataset-size', type=int, default=5000, help='Dataset size to load (default: 5000)')
    
    args = parser.parse_args()
    
    print(f"🚀 Evaluating Model: {args.model}")
    print(f"📊 Dataset Size: {args.dataset_size}")
    print(f"🔬 Evaluation Samples: {args.samples}")
    print("=" * 60)
    
    evaluator = FastModelEvaluator()
    
    # Load dataset
    print("📥 Loading dataset...")
    texts, labels = evaluator.load_dataset_fast(args.dataset_size)
    
    if not texts:
        print("❌ Failed to load dataset")
        return
    
    # Prepare labels
    numeric_labels, label_encoder = evaluator.prepare_labels(labels)
    
    # Evaluate model
    result = evaluator.evaluate_single_model(args.model, texts, numeric_labels, args.samples)
    
    if result:
        # Compare with existing results
        try:
            with open('results/results_summary.json', 'r') as f:
                existing_results = json.load(f)
            
            print(f"\n📊 COMPARISON WITH EXISTING RESULTS:")
            print("-" * 50)
            print(f"{'Model':<20} {'F1-Score':<10} {'Accuracy':<10}")
            print("-" * 50)
            
            # Show new result first
            print(f"{result['model']:<20} {result['f1_score']:<10.4f} {result['accuracy']:<10.4f} ⭐ NEW")
            
            # Show existing results
            for existing in existing_results['final_rankings']['by_f1_score']:
                print(f"{existing['model']:<20} {existing['f1_score']:<10.4f} {existing['accuracy']:<10.4f}")
            
        except FileNotFoundError:
            print("No existing results found for comparison.")
        
        # Save new result
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"results/new_model_{result['model']}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'evaluation_info': {
                    'model': args.model,
                    'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'dataset_samples': args.dataset_size,
                    'evaluation_samples': args.samples
                },
                'results': result
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Quick recommendation
        if result['f1_score'] > 0.80:
            print(f"🏆 EXCELLENT performance! This model scores better than average.")
        elif result['f1_score'] > 0.75:
            print(f"✅ GOOD performance! This model is competitive.")
        else:
            print(f"📊 BASELINE performance. Consider fine-tuning or trying other models.")
            
    else:
        print("❌ Evaluation failed")

if __name__ == "__main__":
    main()
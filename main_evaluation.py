#!/usr/bin/env python3
"""
Main script to evaluate multiple language models on Bangla dataset
Models: Llama 3.1, Qwen 2.5, M10, IndicGemma, Sarban 1
Metrics: Accuracy, Precision, Recall, F1-Score
"""

import os
import sys
import torch
import argparse
from data_loader import BanglaDatasetLoader
from model_evaluator import ModelEvaluator
import json
from datetime import datetime

def check_gpu():
    """Check GPU availability and specs"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        print(f"✓ GPU Count: {gpu_count}")
        return True
    else:
        print("⚠ No GPU available, using CPU (will be slower)")
        return False

def find_dataset():
    """Find dataset file in current directory"""
    possible_files = [
        'dataset.csv', 'data.csv', 'bangla_dataset.csv',
        'dataset.json', 'data.json', 'bangla_dataset.json',
        'train.csv', 'test.csv', 'bangla_data.csv'
    ]
    
    found_files = []
    for file_name in possible_files:
        if os.path.exists(file_name):
            found_files.append(file_name)
    
    return found_files

def main():
    parser = argparse.ArgumentParser(description='Evaluate language models on Bangla dataset')
    parser.add_argument('--dataset', type=str, help='Path to dataset file (CSV or JSON)')
    parser.add_argument('--sample_size', type=int, default=20000, help='Number of samples to use')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()
    
    print("🚀 Starting Bangla Dataset Model Evaluation")
    print("="*60)
    
    # Check GPU
    use_gpu = not args.no_gpu
    gpu_available = check_gpu()
    use_gpu = use_gpu and gpu_available
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Step 1: Load Dataset
    print(f"\n📊 Loading Dataset (Target: {args.sample_size} samples)")
    print("-" * 40)
    
    loader = BanglaDatasetLoader(sample_size=args.sample_size)
    
    # Find dataset file
    if args.dataset:
        dataset_path = args.dataset
    else:
        found_files = find_dataset()
        if found_files:
            dataset_path = found_files[0]
            print(f"Found dataset file: {dataset_path}")
        else:
            print("❌ No dataset file found!")
            print("Please provide dataset using --dataset flag or place one of these files in current directory:")
            print("  - dataset.csv, data.csv, bangla_dataset.csv")
            print("  - dataset.json, data.json, bangla_dataset.json")
            return
    
    # Load dataset
    try:
        if args.dataset and args.dataset.startswith("FOLDER:"):
            # Handle folder structure
            folder_path = args.dataset[7:]  # Remove "FOLDER:" prefix
            dataset = loader.load_from_dataset_folder(folder_path)
        elif args.dataset:
            dataset = loader.auto_detect_and_load(dataset_path)
        else:
            # Try to load from bangla dataset folder automatically
            bangla_dataset_path = r"d:\bangla dataset"
            if os.path.exists(bangla_dataset_path):
                dataset = loader.load_from_dataset_folder(bangla_dataset_path)
                print(f"Automatically loaded from: {bangla_dataset_path}")
            else:
                raise ValueError("No dataset found")
        
        print("✓ Dataset loaded successfully!")
        
        # Show dataset info
        info = loader.get_dataset_info()
        print(f"  - Size: {info['size']} samples")
        print(f"  - Columns: {info['columns']}")
        if 'num_classes' in info:
            print(f"  - Classes: {info['num_classes']}")
            print(f"  - Distribution: {info['label_distribution']}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Split dataset
    train_dataset, test_dataset = loader.get_train_test_split(test_size=0.2)
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    
    # Step 2: Initialize Model Evaluator
    print(f"\n🤖 Initializing Model Evaluator")
    print("-" * 40)
    
    evaluator = ModelEvaluator(use_gpu=use_gpu, use_4bit=True)
    
    # Step 3: Evaluate Models
    print(f"\n🔥 Starting Model Evaluation")
    print("-" * 40)
    print("Models to evaluate:")
    print("  1. Llama 3.1")
    print("  2. Qwen 2.5") 
    print("  3. IndicGemma")
    print("  4. M10 (if available)")
    print("  5. Sarban 1 (if available)")
    
    # Run evaluation
    results = evaluator.evaluate_all_models(test_dataset)
    
    # Step 4: Save and Display Results
    print(f"\n💾 Saving Results")
    print("-" * 40)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output, f"evaluation_results_{timestamp}.json")
    evaluator.save_results(results, results_file)
    
    # Create and save summary
    summary = evaluator.create_summary_report(results)
    
    summary_file = os.path.join(args.output, f"summary_report_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("BANGLA DATASET MODEL EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total samples used: {args.sample_size}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}\n")
        f.write("-" * 70 + "\n")
        
        for result in summary:
            f.write(f"{result['model']:<20} "
                   f"{result['accuracy']:<10.4f} "
                   f"{result['precision']:<12.4f} "
                   f"{result['recall']:<10.4f} "
                   f"{result['f1_score']:<10.4f}\n")
    
    print(f"✓ Detailed results saved to: {results_file}")
    print(f"✓ Summary report saved to: {summary_file}")
    
    print(f"\n🎉 Evaluation Complete!")
    print(f"Check the {args.output} directory for detailed results.")

if __name__ == "__main__":
    main()
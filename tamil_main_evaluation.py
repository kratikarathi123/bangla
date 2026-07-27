#!/usr/bin/env python3
"""
============================================================================
ULTIMATE TAMIL LANGUAGE MODEL EVALUATION
============================================================================
Evaluates 20 models across 5 categories on Tamil text classification:

Traditional ML (4):    Naive Bayes, SVM, Logistic Regression, Random Forest
Ensemble (1):          XGBoost
Deep Learning (5):     CNN, LSTM, Bi-LSTM, GRU, CNN-LSTM
Transformers (5):      BERT, XLM-RoBERTa, IndicBERT v2, MuRIL, RemBERT
LLMs (5):              IndicGemma, Llama 3.1, Qwen 2.5, Sarban 1, M10

Dataset: Tamil news article classification (IndicNLP / Hugging Face)
Metrics: Accuracy, Precision, Recall, F1-Score
============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Local imports
from tamil_data_loader import TamilDatasetLoader
from tamil_traditional_models import TamilTraditionalEvaluator
from tamil_transformer_models import TamilTransformerEvaluator


def print_banner():
    """Print evaluation banner"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        ULTIMATE TAMIL LANGUAGE MODEL EVALUATION                 ║
║        20 Models × 5 Categories × Tamil Text Classification     ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def run_evaluation(sample_size=20000, skip_llms=False, skip_transformers=False):
    """Run the complete Tamil model evaluation pipeline"""
    
    print_banner()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================
    # STEP 1: Load Dataset
    # ========================
    print("\n" + "=" * 70)
    print("📊 STEP 1: Loading Tamil Dataset")
    print("=" * 70)
    
    loader = TamilDatasetLoader(sample_size=sample_size)
    dataset = loader.load_indicnlp_tamil()
    info = loader.get_dataset_info()
    
    print(f"\n   Dataset Summary:")
    print(f"   • Total samples: {info['size']}")
    print(f"   • Classes: {info['num_classes']}")
    print(f"   • Categories: {info['label_names']}")
    print(f"   • Distribution: {info['label_distribution']}")
    
    # Get data
    texts, labels = loader.get_texts_and_labels()
    label_names = info['label_names']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        list(texts), list(labels), test_size=0.2, random_state=42, stratify=list(labels)
    )
    
    print(f"\n   Train/Test Split:")
    print(f"   • Training: {len(X_train)} samples")
    print(f"   • Testing:  {len(X_test)} samples")
    
    all_results = []
    
    # ========================
    # STEP 2: Traditional ML + Deep Learning
    # ========================
    print("\n" + "=" * 70)
    print("🔧 STEP 2: Traditional ML & Deep Learning Models")
    print("=" * 70)
    
    trad_evaluator = TamilTraditionalEvaluator()
    trad_results = trad_evaluator.evaluate_all(X_train, X_test, y_train, y_test)
    all_results.extend(trad_results)
    
    # ========================
    # STEP 3: Transformers
    # ========================
    if not skip_transformers:
        print("\n" + "=" * 70)
        print("🤖 STEP 3: Transformer Models")
        print("=" * 70)
        
        trans_evaluator = TamilTransformerEvaluator()
        trans_results = trans_evaluator.evaluate_all_transformers(X_train, X_test, y_train, y_test)
        all_results.extend(trans_results)
    else:
        print("\n⏩ Skipping Transformer models (--skip-transformers flag)")
    
    # ========================
    # STEP 4: LLMs
    # ========================
    if not skip_llms:
        print("\n" + "=" * 70)
        print("🧠 STEP 4: Large Language Models")
        print("=" * 70)
        
        if skip_transformers:
            trans_evaluator = TamilTransformerEvaluator()
        
        llm_results = trans_evaluator.evaluate_all_llms(X_test, y_test, label_names)
        all_results.extend(llm_results)
    else:
        print("\n⏩ Skipping LLM models (--skip-llms flag)")
    
    # ========================
    # STEP 5: Results & Report
    # ========================
    print("\n" + "=" * 70)
    print("📋 STEP 5: Final Results & Report Generation")
    print("=" * 70)
    
    # Filter out failed models
    valid_results = [r for r in all_results if 'error' not in r and r['f1_score'] > 0]
    
    # Sort by F1
    valid_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Display final rankings
    print(f"\n{'='*90}")
    print("🏆 FINAL RANKINGS - Tamil Text Classification")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Model':<20} {'Category':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 97)
    
    for i, r in enumerate(valid_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}"
        print(f"{medal:<5} {r['model']:<20} {r['category']:<22} "
              f"{r['accuracy']:<10.4f} {r['precision']:<10.4f} "
              f"{r['recall']:<10.4f} {r['f1_score']:<10.4f}")
    
    # ========================
    # SAVE RESULTS
    # ========================
    os.makedirs('results', exist_ok=True)
    
    # Save JSON
    json_file = f"results/tamil_evaluation_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_date': datetime.now().isoformat(),
            'language': 'Tamil',
            'dataset_info': {
                'total_samples': info['size'],
                'num_classes': info['num_classes'],
                'categories': [str(c) for c in info['label_names']],
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'results': all_results,
            'rankings': valid_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved: {json_file}")
    
    # Save CSV
    csv_file = f"results/tamil_comparison_{timestamp}.csv"
    df = pd.DataFrame(valid_results)
    df.insert(0, 'rank', range(1, len(df)+1))
    df.to_csv(csv_file, index=False)
    print(f"💾 CSV saved: {csv_file}")
    
    # Generate report
    report_file = f"results/TAMIL_EVALUATION_REPORT.md"
    _generate_report(valid_results, info, report_file, timestamp)
    print(f"💾 Report saved: {report_file}")
    
    # Summary
    if valid_results:
        print(f"\n🎉 Evaluation Complete!")
        print(f"   Champion: {valid_results[0]['model']} (F1: {valid_results[0]['f1_score']:.4f})")
        print(f"   Total models evaluated: {len(valid_results)}")
    
    return all_results


def _generate_report(results, info, filepath, timestamp):
    """Generate a markdown report"""
    
    report = f"""# Tamil Language Model Evaluation - Complete Results

## Executive Summary

- **Language**: Tamil (தமிழ்)
- **Total Models Evaluated**: {len(results)} models across multiple categories
- **Dataset**: {info['size']} Tamil text samples, {info['num_classes']} categories
- **Evaluation Date**: {datetime.now().strftime('%B %d, %Y')}
- **Champion**: **{results[0]['model']}** ({results[0]['f1_score']*100:.2f}% F1-Score)

---

## Complete Final Rankings

| Rank | Model | Category | Accuracy | Precision | Recall | F1-Score | Training Time |
|------|-------|----------|----------|-----------|--------|----------|---------------|
"""
    
    for i, r in enumerate(results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
        report += (f"| {medal} | **{r['model']}** | {r['category']} | "
                   f"{r['accuracy']*100:.2f}% | {r['precision']*100:.2f}% | "
                   f"{r['recall']*100:.2f}% | {r['f1_score']*100:.2f}% | "
                   f"{r['training_time']:.1f}s |\n")
    
    report += f"""
---

## Category Champions

"""
    
    # Group by category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    for cat, models in categories.items():
        best = models[0]
        avg_f1 = np.mean([m['f1_score'] for m in models])
        report += f"""### {cat} ({len(models)} models)
- **Champion**: {best['model']} (F1: {best['f1_score']*100:.2f}%)
- **Average F1**: {avg_f1*100:.2f}%

"""
    
    report += f"""---

## Dataset Details

- **Language**: Tamil (தமிழ்) - Dravidian language family
- **Size**: {info['size']} text samples
- **Categories**: {info['num_classes']} classes
- **Category Names**: {', '.join([str(c) for c in info['label_names']])}
- **Split**: 80% training, 20% testing
- **Encoding**: UTF-8 for proper Tamil character support

---

## Technical Specifications

- **Framework**: PyTorch, TensorFlow, Scikit-learn
- **Evaluation Methodology**: Accuracy, Precision, Recall, F1-Score (weighted)
- **Cross-validation**: Stratified train/test split
- **Random Seed**: 42 (for reproducibility)

---

## Comparison with Bangla Results

This evaluation mirrors the Bangla language evaluation methodology, enabling direct
cross-lingual comparison between Tamil (Dravidian family) and Bangla (Indo-Aryan family).

Key differences:
- Tamil uses a different script (Tamil script vs Bengali script)
- Tamil belongs to a different language family (Dravidian vs Indo-Aryan)
- Tamil has agglutinative morphology, which may affect tokenization

---

*Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Evaluation Framework: Tamil Multi-Paradigm Pipeline*
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tamil Language Model Evaluation')
    parser.add_argument('--sample-size', type=int, default=20000,
                        help='Number of samples to use (default: 20000)')
    parser.add_argument('--skip-llms', action='store_true',
                        help='Skip LLM evaluation (requires large GPU)')
    parser.add_argument('--skip-transformers', action='store_true',
                        help='Skip transformer evaluation')
    parser.add_argument('--traditional-only', action='store_true',
                        help='Only run traditional ML + deep learning models')
    
    args = parser.parse_args()
    
    skip_llms = args.skip_llms or args.traditional_only
    skip_transformers = args.skip_transformers or args.traditional_only
    
    # Default: skip LLMs since they require huge downloads (7B models)
    if not args.skip_llms and not args.traditional_only:
        print("💡 Note: LLM models (7B params) require large downloads. Use --skip-llms to skip them.")
        print("   Running LLMs by default. Press Ctrl+C to cancel if downloads are too slow.\n")
    
    results = run_evaluation(
        sample_size=args.sample_size,
        skip_llms=skip_llms,
        skip_transformers=skip_transformers
    )

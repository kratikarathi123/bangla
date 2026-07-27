#!/usr/bin/env python3
"""
Re-check Tamil Models with Cross-Validation to detect overfitting.
Uses ALL available data (train + val + test = 6,684 samples).
Adds 5-fold stratified cross-validation for SVM and other models.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import time
import json

print("=" * 70)
print("  TAMIL MODEL RE-CHECK WITH CROSS-VALIDATION")
print("  Using ALL available data (6,684 samples)")
print("=" * 70)

# Load ALL splits
print("\n[1/4] Loading all data splits...")
ds = load_dataset('indic_glue', 'inltkh.ta')
train_ds = ds['train']
val_ds = ds['validation']
test_ds = ds['test']

# Combine all data
all_data = concatenate_datasets([train_ds, val_ds, test_ds])
print(f"  Total samples: {len(all_data)}")
print(f"  Labels: {set(all_data['label'])}")

texts = all_data['text']
labels = np.array(all_data['label'])

# Remap labels to 0-indexed
unique_labels = sorted(set(labels))
label_map = {old: new for new, old in enumerate(unique_labels)}
labels = np.array([label_map[l] for l in labels])
print(f"  Classes: {len(unique_labels)} -> remapped to {list(range(len(unique_labels)))}")
print(f"  Distribution: {np.bincount(labels)}")

# TF-IDF Features
print("\n[2/4] Extracting TF-IDF features (char n-grams 1-3)...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    analyzer='char_wb',
    sublinear_tf=True
)
X_tfidf = vectorizer.fit_transform(texts)
print(f"  Feature matrix: {X_tfidf.shape}")

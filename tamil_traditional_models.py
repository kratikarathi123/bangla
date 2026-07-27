#!/usr/bin/env python3
"""
Traditional ML and Deep Learning Models for Tamil Text Classification
All deep learning models use PyTorch with CUDA GPU support.

Implements: Naive Bayes, SVM, Logistic Regression, Random Forest, XGBoost,
            CNN, LSTM, Bi-LSTM, GRU, CNN-LSTM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import xgboost as xgb
from tqdm import tqdm
import time
import gc


# ========================
# PyTorch Dataset
# ========================

class TamilTextSequenceDataset(Dataset):
    """PyTorch Dataset for sequence models (CNN, LSTM, etc.)"""
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ========================
# PyTorch Deep Learning Models
# ========================

class CNNClassifier(nn.Module):
    """CNN for text classification"""
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=128, filter_sizes=[3, 4, 5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outs = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class LSTMClassifier(nn.Module):
    """LSTM for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        return self.fc(x)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.bilstm(x)
        x = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = self.dropout(x)
        return self.fc(x)


class GRUClassifier(nn.Module):
    """GRU for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        x = self.dropout(hidden[-1])
        return self.fc(x)


class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM hybrid for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(50)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv(x))            # (batch, 64, seq_len)
        x = self.pool(x)                        # (batch, 64, 50)
        x = x.transpose(1, 2)                   # (batch, 50, 64)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        return self.fc(x)


# ========================
# Simple Tokenizer
# ========================

class SimpleTokenizer:
    """Simple word-level tokenizer for Tamil text"""
    def __init__(self, max_words=10000, max_length=200):
        self.max_words = max_words
        self.max_length = max_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
    
    def fit(self, texts):
        """Build vocabulary from texts"""
        for text in texts:
            for word in str(text).split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Keep top max_words
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[:self.max_words - 2]):
            idx = i + 2
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def texts_to_sequences(self, texts):
        """Convert texts to padded sequences"""
        sequences = []
        for text in texts:
            words = str(text).split()
            seq = [self.word2idx.get(w, 1) for w in words[:self.max_length]]
            # Pad
            if len(seq) < self.max_length:
                seq = seq + [0] * (self.max_length - len(seq))
            sequences.append(seq)
        return np.array(sequences)


# ========================
# Main Evaluator
# ========================

class TamilTraditionalEvaluator:
    """Evaluator for Traditional ML and Deep Learning models on Tamil text"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Tamil Traditional Model Evaluator")
        print(f"   Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        torch.manual_seed(42)
        np.random.seed(42)
    
    def prepare_tfidf_features(self, X_train, X_test):
        """Prepare TF-IDF features for traditional ML models"""
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            analyzer='char_wb',
            sublinear_tf=True
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        return X_train_vec, X_test_vec, vectorizer
    
    def _prepare_dl_data(self, X_train, X_test, y_train, y_test, batch_size=64):
        """Prepare data for PyTorch deep learning models"""
        tokenizer = SimpleTokenizer(max_words=10000, max_length=200)
        tokenizer.fit(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        train_dataset = TamilTextSequenceDataset(X_train_seq, y_train)
        test_dataset = TamilTextSequenceDataset(X_test_seq, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader, tokenizer
    
    def _train_and_eval_dl(self, model, train_loader, test_loader, epochs=5, lr=0.001):
        """Train and evaluate a PyTorch DL model"""
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        # Training
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step(total_loss)
        
        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                outputs = model(sequences)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return true_labels, predictions
    
    # ========================
    # TRADITIONAL ML MODELS
    # ========================
    
    def evaluate_naive_bayes(self, X_train, X_test, y_train, y_test):
        """Evaluate Naive Bayes classifier"""
        print("\n🧠 Evaluating Naive Bayes...")
        start_time = time.time()
        
        X_train_vec, X_test_vec, _ = self.prepare_tfidf_features(X_train, X_test)
        
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'Naive Bayes',
            'category': 'Traditional ML',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    def evaluate_svm(self, X_train, X_test, y_train, y_test):
        """Evaluate SVM classifier"""
        print("\n⚖️  Evaluating SVM (RBF)...")
        start_time = time.time()
        
        X_train_vec, X_test_vec, _ = self.prepare_tfidf_features(X_train, X_test)
        
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'SVM (RBF)',
            'category': 'Traditional ML',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    def evaluate_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Evaluate Logistic Regression"""
        print("\n📊 Evaluating Logistic Regression...")
        start_time = time.time()
        
        X_train_vec, X_test_vec, _ = self.prepare_tfidf_features(X_train, X_test)
        
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, multi_class='ovr')
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'Logistic Regression',
            'category': 'Traditional ML',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    def evaluate_random_forest(self, X_train, X_test, y_train, y_test):
        """Evaluate Random Forest"""
        print("\n🌳 Evaluating Random Forest...")
        start_time = time.time()
        
        X_train_vec, X_test_vec, _ = self.prepare_tfidf_features(X_train, X_test)
        
        model = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'Random Forest',
            'category': 'Traditional ML',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    def evaluate_xgboost(self, X_train, X_test, y_train, y_test):
        """Evaluate XGBoost"""
        print("\n🚀 Evaluating XGBoost...")
        start_time = time.time()
        
        X_train_vec, X_test_vec, _ = self.prepare_tfidf_features(X_train, X_test)
        
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=42,
            eval_metric='mlogloss', device='cuda'
        )
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'XGBoost',
            'category': 'Ensemble Learning',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    # ========================
    # DEEP LEARNING MODELS (PyTorch + CUDA)
    # ========================
    
    def evaluate_cnn(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate CNN model on GPU"""
        print("\n🔥 Evaluating CNN (PyTorch + CUDA)...")
        start_time = time.time()
        
        num_classes = len(set(y_train))
        train_loader, test_loader, tokenizer = self._prepare_dl_data(X_train, X_test, y_train, y_test)
        
        model = CNNClassifier(
            vocab_size=10000, embed_dim=128,
            num_classes=num_classes, num_filters=128
        )
        
        true_labels, predictions = self._train_and_eval_dl(model, train_loader, test_loader, epochs=epochs)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'CNN',
            'category': 'Deep Learning',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    def evaluate_lstm(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate LSTM model on GPU"""
        print("\n🔄 Evaluating LSTM (PyTorch + CUDA)...")
        start_time = time.time()
        
        num_classes = len(set(y_train))
        train_loader, test_loader, _ = self._prepare_dl_data(X_train, X_test, y_train, y_test)
        
        model = LSTMClassifier(
            vocab_size=10000, embed_dim=128,
            hidden_dim=64, num_classes=num_classes
        )
        
        true_labels, predictions = self._train_and_eval_dl(model, train_loader, test_loader, epochs=epochs)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'LSTM',
            'category': 'Deep Learning',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    def evaluate_bilstm(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate Bidirectional LSTM model on GPU"""
        print("\n🔄🔄 Evaluating Bi-LSTM (PyTorch + CUDA)...")
        start_time = time.time()
        
        num_classes = len(set(y_train))
        train_loader, test_loader, _ = self._prepare_dl_data(X_train, X_test, y_train, y_test)
        
        model = BiLSTMClassifier(
            vocab_size=10000, embed_dim=128,
            hidden_dim=64, num_classes=num_classes
        )
        
        true_labels, predictions = self._train_and_eval_dl(model, train_loader, test_loader, epochs=epochs)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'Bi-LSTM',
            'category': 'Deep Learning',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    def evaluate_gru(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate GRU model on GPU"""
        print("\n🌀 Evaluating GRU (PyTorch + CUDA)...")
        start_time = time.time()
        
        num_classes = len(set(y_train))
        train_loader, test_loader, _ = self._prepare_dl_data(X_train, X_test, y_train, y_test)
        
        model = GRUClassifier(
            vocab_size=10000, embed_dim=128,
            hidden_dim=64, num_classes=num_classes
        )
        
        true_labels, predictions = self._train_and_eval_dl(model, train_loader, test_loader, epochs=epochs)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'GRU',
            'category': 'Deep Learning',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    def evaluate_cnn_lstm(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate CNN-LSTM hybrid model on GPU"""
        print("\n🔥🔄 Evaluating CNN-LSTM (PyTorch + CUDA)...")
        start_time = time.time()
        
        num_classes = len(set(y_train))
        train_loader, test_loader, _ = self._prepare_dl_data(X_train, X_test, y_train, y_test)
        
        model = CNNLSTMClassifier(
            vocab_size=10000, embed_dim=128,
            hidden_dim=64, num_classes=num_classes
        )
        
        true_labels, predictions = self._train_and_eval_dl(model, train_loader, test_loader, epochs=epochs)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        elapsed = time.time() - start_time
        
        return {
            'model': 'CNN-LSTM',
            'category': 'Deep Learning',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed
        }
    
    # ========================
    # RUN ALL
    # ========================
    
    def evaluate_all(self, X_train, X_test, y_train, y_test):
        """Evaluate all traditional ML and deep learning models"""
        print("\n" + "=" * 70)
        print("🚀 EVALUATING ALL TRADITIONAL ML & DEEP LEARNING MODELS")
        print("=" * 70)
        
        results = []
        
        # Traditional ML
        models = [
            ('Naive Bayes', self.evaluate_naive_bayes),
            ('SVM', self.evaluate_svm),
            ('Logistic Regression', self.evaluate_logistic_regression),
            ('Random Forest', self.evaluate_random_forest),
            ('XGBoost', self.evaluate_xgboost),
        ]
        
        for name, func in models:
            try:
                result = func(X_train, X_test, y_train, y_test)
                results.append(result)
                print(f"   ✅ {result['model']}: F1={result['f1_score']:.4f}, Time={result['training_time']:.1f}s")
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        # Deep Learning (PyTorch + CUDA)
        dl_models = [
            ('CNN', self.evaluate_cnn),
            ('LSTM', self.evaluate_lstm),
            ('Bi-LSTM', self.evaluate_bilstm),
            ('GRU', self.evaluate_gru),
            ('CNN-LSTM', self.evaluate_cnn_lstm),
        ]
        
        for name, func in dl_models:
            try:
                # LSTM/GRU/CNN-LSTM need more epochs to converge
                ep = 15 if name in ('LSTM', 'GRU', 'CNN-LSTM') else 5
                result = func(X_train, X_test, y_train, y_test, epochs=ep)
                results.append(result)
                print(f"   ✅ {result['model']}: F1={result['f1_score']:.4f}, Time={result['training_time']:.1f}s")
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        return results


if __name__ == "__main__":
    from tamil_data_loader import TamilDatasetLoader
    
    print("🇮🇳 Tamil Traditional & Deep Learning Model Evaluation")
    print("=" * 60)
    
    # Load data
    loader = TamilDatasetLoader(sample_size=20000)
    dataset = loader.load_indicnlp_tamil()
    
    texts, labels = loader.get_texts_and_labels()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        list(texts), list(labels), test_size=0.2, random_state=42, stratify=list(labels)
    )
    
    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Evaluate
    evaluator = TamilTraditionalEvaluator()
    results = evaluator.evaluate_all(X_train, X_test, y_train, y_test)
    
    # Display results
    print(f"\n{'='*80}")
    print("🏆 RESULTS - Traditional ML & Deep Learning")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Category':<18} {'Accuracy':<10} {'F1-Score':<10} {'Time(s)':<10}")
    print("-" * 78)
    
    for r in sorted(results, key=lambda x: x['f1_score'], reverse=True):
        print(f"{r['model']:<20} {r['category']:<18} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f} {r['training_time']:<10.1f}")

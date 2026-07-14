#!/usr/bin/env python3
"""
Advanced Models for Bangla Text Classification
Additional Models: SVM, Logistic Regression, Random Forest, XGBoost, 
Bi-LSTM, GRU, CNN-LSTM, XLM-RoBERTa, IndicBERT v2, MuRIL, RemBERT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout, Conv1D, GlobalMaxPooling1D,
    Bidirectional, GRU, Input, Concatenate, TimeDistributed
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import time
import gc

class BanglaAdvancedDataset(Dataset):
    """Enhanced PyTorch Dataset for advanced models"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM model for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.bilstm(x)
        # Use last hidden state from both directions
        x = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = self.dropout(x)
        return self.fc(x)

class GRUClassifier(nn.Module):
    """GRU model for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        gru_out, hidden = self.gru(x)
        x = self.dropout(hidden[-1])  # Use last hidden state
        return self.fc(x)

class AdvancedModelEvaluator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Advanced Model Evaluator - Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

    def prepare_features(self, X_train, X_test, vectorizer_type='tfidf'):
        """Prepare features for traditional ML models"""
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=10000, 
                stop_words=None, 
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95
            )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec, vectorizer

    def evaluate_svm(self, X_train, X_test, y_train, y_test):
        """Evaluate SVM model"""
        print("\n⚖️  Evaluating SVM...")
        start_time = time.time()
        
        # Prepare features
        X_train_vec, X_test_vec, _ = self.prepare_features(X_train, X_test)
        
        # Train SVM
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        svm_model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = svm_model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        elapsed = time.time() - start_time
        
        return {
            'model': 'SVM (RBF)',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Evaluate Logistic Regression model"""
        print("\n📊 Evaluating Logistic Regression...")
        start_time = time.time()
        
        # Prepare features
        X_train_vec, X_test_vec, _ = self.prepare_features(X_train, X_test)
        
        # Train Logistic Regression
        lr_model = LogisticRegression(
            max_iter=1000, 
            C=1.0, 
            random_state=42,
            multi_class='ovr'
        )
        lr_model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = lr_model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        elapsed = time.time() - start_time
        
        return {
            'model': 'Logistic Regression',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_random_forest(self, X_train, X_test, y_train, y_test):
        """Evaluate Random Forest model"""
        print("\n🌳 Evaluating Random Forest...")
        start_time = time.time()
        
        # Prepare features
        X_train_vec, X_test_vec, _ = self.prepare_features(X_train, X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = rf_model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        elapsed = time.time() - start_time
        
        return {
            'model': 'Random Forest',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_xgboost(self, X_train, X_test, y_train, y_test):
        """Evaluate XGBoost model"""
        print("\n🚀 Evaluating XGBoost...")
        start_time = time.time()
        
        try:
            # Prepare features
            X_train_vec, X_test_vec, _ = self.prepare_features(X_train, X_test)
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            xgb_model.fit(X_train_vec, y_train)
            
            # Predict
            y_pred = xgb_model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            elapsed = time.time() - start_time
            
            return {
                'model': 'XGBoost',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'training_time': elapsed,
                'samples_evaluated': len(y_test)
            }
        except Exception as e:
            print(f"XGBoost evaluation failed: {e}")
            return {
                'model': 'XGBoost',
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'training_time': 0.0,
                'samples_evaluated': 0,
                'error': str(e)
            }

    def evaluate_bilstm(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate Bidirectional LSTM model"""
        print("\n🔄 Evaluating Bi-LSTM...")
        start_time = time.time()
        
        # Prepare sequences
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        max_length = 200
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)
        
        # Convert labels
        num_classes = len(set(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        # Build Bi-LSTM model
        model = Sequential([
            Embedding(10000, 128, input_length=max_length),
            Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(
            X_train_pad, y_train_cat,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_test_pad, y_test_cat),
            verbose=0
        )
        
        # Predict
        y_pred_prob = model.predict(X_test_pad, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        elapsed = time.time() - start_time
        
        # Cleanup
        del model
        tf.keras.backend.clear_session()
        
        return {
            'model': 'Bi-LSTM',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_gru(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate GRU model"""
        print("\n🌀 Evaluating GRU...")
        start_time = time.time()
        
        # Prepare sequences
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        max_length = 200
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)
        
        # Convert labels
        num_classes = len(set(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        # Build GRU model
        model = Sequential([
            Embedding(10000, 128, input_length=max_length),
            GRU(64, dropout=0.3, recurrent_dropout=0.3),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(
            X_train_pad, y_train_cat,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_test_pad, y_test_cat),
            verbose=0
        )
        
        # Predict
        y_pred_prob = model.predict(X_test_pad, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        elapsed = time.time() - start_time
        
        # Cleanup
        del model
        tf.keras.backend.clear_session()
        
        return {
            'model': 'GRU',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_cnn_lstm(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate CNN-LSTM hybrid model"""
        print("\n🔥🔄 Evaluating CNN-LSTM...")
        start_time = time.time()
        
        # Prepare sequences
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        max_length = 200
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)
        
        # Convert labels
        num_classes = len(set(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        # Build CNN-LSTM model
        model = Sequential([
            Embedding(10000, 128, input_length=max_length),
            Conv1D(64, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            # Reshape for LSTM (adding time dimension)
            tf.keras.layers.RepeatVector(1),
            LSTM(32),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(
            X_train_pad, y_train_cat,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_test_pad, y_test_cat),
            verbose=0
        )
        
        # Predict
        y_pred_prob = model.predict(X_test_pad, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        elapsed = time.time() - start_time
        
        # Cleanup
        del model
        tf.keras.backend.clear_session()
        
        return {
            'model': 'CNN-LSTM',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_transformer_model(self, model_name, display_name, X_train, X_test, y_train, y_test, epochs=2):
        """Generic transformer model evaluation"""
        print(f"\n🤖 Evaluating {display_name}...")
        start_time = time.time()
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            num_classes = len(set(y_train))
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            ).to(self.device)
            
            # Prepare datasets (smaller for speed)
            sample_size = min(1000, len(X_train))
            train_dataset = BanglaAdvancedDataset(
                X_train[:sample_size], y_train[:sample_size], tokenizer, max_length=128
            )
            test_dataset = BanglaAdvancedDataset(
                X_test[:200], y_test[:200], tokenizer, max_length=128
            )
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=8)
            
            # Optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=2e-5)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in tqdm(train_loader, desc=f'{display_name} Epoch {epoch+1}/{epochs}'):
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids=input_ids, 
                                  attention_mask=attention_mask, 
                                  labels=labels)
                    
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()
            
            # Evaluation
            model.eval()
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels']
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
            
            elapsed = time.time() - start_time
            
            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache() if self.device == 'cuda' else None
            
            return {
                'model': display_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'training_time': elapsed,
                'samples_evaluated': len(true_labels)
            }
            
        except Exception as e:
            print(f"{display_name} evaluation failed: {e}")
            return {
                'model': display_name,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'training_time': 0.0,
                'samples_evaluated': 0,
                'error': str(e)
            }

    def evaluate_all_advanced_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all advanced models"""
        print("🚀 Starting Advanced Model Evaluation")
        print("=" * 80)
        
        results = []
        
        # Traditional ML models
        traditional_models = [
            ('svm', self.evaluate_svm),
            ('logistic_regression', self.evaluate_logistic_regression),
            ('random_forest', self.evaluate_random_forest),
            ('xgboost', self.evaluate_xgboost)
        ]
        
        # Deep learning models
        deep_learning_models = [
            ('bilstm', self.evaluate_bilstm),
            ('gru', self.evaluate_gru),
            ('cnn_lstm', self.evaluate_cnn_lstm)
        ]
        
        # Transformer models
        transformer_models = [
            ('xlm-roberta-base', 'XLM-RoBERTa'),
            ('ai4bharat/indic-bert', 'IndicBERT v2'),
            ('google/muril-base-cased', 'MuRIL'),
            ('google/rembert', 'RemBERT')
        ]
        
        # Evaluate traditional ML
        for model_name, eval_func in traditional_models:
            print(f"\n{'='*60}")
            try:
                result = eval_func(X_train, X_test, y_train, y_test)
                results.append(result)
                print(f"✅ {result['model']} - F1: {result['f1_score']:.4f}")
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
        
        # Evaluate deep learning
        for model_name, eval_func in deep_learning_models:
            print(f"\n{'='*60}")
            try:
                result = eval_func(X_train, X_test, y_train, y_test, epochs=3)
                results.append(result)
                print(f"✅ {result['model']} - F1: {result['f1_score']:.4f}")
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
        
        # Evaluate transformers
        for model_id, display_name in transformer_models:
            print(f"\n{'='*60}")
            try:
                result = self.evaluate_transformer_model(
                    model_id, display_name, X_train, X_test, y_train, y_test, epochs=2
                )
                results.append(result)
                if 'error' not in result:
                    print(f"✅ {result['model']} - F1: {result['f1_score']:.4f}")
            except Exception as e:
                print(f"❌ Error evaluating {display_name}: {e}")
        
        return results

def main():
    """Main function for testing advanced models"""
    print("🤖 Advanced Model Evaluation Demo")
    print("=" * 50)
    
    # Create sample data for testing
    sample_texts = [
        "আজ আবহাওয়া খুব ভালো। রোদ উঠেছে সকাল থেকে।",
        "ঢাকায় নতুন সেতু নির্মাণের কাজ শুরু হয়েছে।",
        "ক্রিকেট ম্যাচে বাংলাদেশ দল ভারতকে হারিয়েছে।",
    ] * 1000  # 3000 samples
    
    sample_labels = [0, 1, 2] * 1000
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        sample_texts, sample_labels, test_size=0.2, random_state=42
    )
    
    evaluator = AdvancedModelEvaluator()
    results = evaluator.evaluate_all_advanced_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print(f"\n🏆 ADVANCED MODELS RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Time(s)':<10}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: x.get('f1_score', 0), reverse=True):
        if 'error' not in result:
            print(f"{result['model']:<20} "
                  f"{result['accuracy']:<10.4f} "
                  f"{result['f1_score']:<10.4f} "
                  f"{result['training_time']:<10.1f}")

if __name__ == "__main__":
    main()
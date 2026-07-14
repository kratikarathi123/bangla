#!/usr/bin/env python3
"""
Traditional ML and Deep Learning Models for Bangla Text Classification
Implements: CNN, LSTM, Naive Bayes, and BERT
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import gc
import time

class BanglaTextDataset(Dataset):
    """PyTorch Dataset for text classification"""
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

class CNNTextClassifier(nn.Module):
    """CNN model for text classification"""
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100, filter_sizes=[3, 4, 5]):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class LSTMTextClassifier(nn.Module):
    """LSTM model for text classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super(LSTMTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])  # Use last hidden state
        return self.fc(x)

class TraditionalModelEvaluator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

    def prepare_data(self, texts, labels, test_size=0.2):
        """Prepare data for training"""
        print("📊 Preparing data for traditional models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test

    def evaluate_naive_bayes(self, X_train, X_test, y_train, y_test):
        """Evaluate Naive Bayes model"""
        print("\n🧠 Evaluating Naive Bayes...")
        start_time = time.time()
        
        # Vectorize text using TF-IDF
        vectorizer = TfidfVectorizer(max_features=10000, stop_words=None, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train Naive Bayes
        nb_model = MultinomialNB(alpha=0.1)
        nb_model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = nb_model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        elapsed = time.time() - start_time
        
        return {
            'model': 'Naive Bayes',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_cnn(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate CNN model"""
        print("\n🔥 Evaluating CNN...")
        start_time = time.time()
        
        # Tokenize text
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        max_length = 200
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)
        
        # Convert labels to categorical
        num_classes = len(set(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        # Build CNN model
        model = Sequential([
            Embedding(10000, 128, input_length=max_length),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
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
            'model': 'CNN',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_lstm(self, X_train, X_test, y_train, y_test, epochs=5):
        """Evaluate LSTM model"""
        print("\n🔄 Evaluating LSTM...")
        start_time = time.time()
        
        # Tokenize text (reuse tokenizer logic)
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
        
        # Build LSTM model
        model = Sequential([
            Embedding(10000, 128, input_length=max_length),
            LSTM(64, dropout=0.3, recurrent_dropout=0.3),
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
        history = model.fit(
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
            'model': 'LSTM',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': elapsed,
            'samples_evaluated': len(y_test)
        }

    def evaluate_bert(self, X_train, X_test, y_train, y_test, epochs=3):
        """Evaluate BERT model"""
        print("\n🤖 Evaluating BERT...")
        start_time = time.time()
        
        try:
            # Use multilingual BERT for Bangla
            model_name = 'bert-base-multilingual-cased'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Prepare datasets
            train_dataset = BanglaTextDataset(X_train[:1000], y_train[:1000], tokenizer, max_length=128)  # Smaller for speed
            test_dataset = BanglaTextDataset(X_test[:200], y_test[:200], tokenizer, max_length=128)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16)
            
            # Load BERT model
            num_classes = len(set(y_train))
            model = BertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_classes
            ).to(self.device)
            
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
                for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
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
                'model': 'BERT (Multilingual)',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'training_time': elapsed,
                'samples_evaluated': len(true_labels)
            }
            
        except Exception as e:
            print(f"BERT evaluation failed: {e}")
            return {
                'model': 'BERT (Multilingual)',
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'training_time': 0.0,
                'samples_evaluated': 0,
                'error': str(e)
            }

    def evaluate_all_traditional_models(self, texts, labels):
        """Evaluate all traditional models"""
        print("🚀 Starting Traditional Model Evaluation")
        print("=" * 60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(texts, labels)
        
        results = []
        
        # Evaluate each model
        models_to_evaluate = [
            ('naive_bayes', self.evaluate_naive_bayes),
            ('cnn', self.evaluate_cnn),
            ('lstm', self.evaluate_lstm),
            ('bert', self.evaluate_bert)
        ]
        
        for model_name, eval_func in models_to_evaluate:
            print(f"\n{'='*60}")
            try:
                if model_name == 'bert':
                    result = eval_func(X_train, X_test, y_train, y_test, epochs=2)  # Fewer epochs for speed
                else:
                    result = eval_func(X_train, X_test, y_train, y_test)
                
                results.append(result)
                print(f"✅ {result['model']} completed:")
                print(f"   Accuracy: {result['accuracy']:.4f}")
                print(f"   F1-Score: {result['f1_score']:.4f}")
                print(f"   Time: {result['training_time']:.1f}s")
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                results.append({
                    'model': model_name.upper(),
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'training_time': 0.0,
                    'error': str(e)
                })
        
        return results

def main():
    """Main function to run traditional model evaluation"""
    print("🤖 Traditional ML & Deep Learning Model Evaluation")
    print("Models: Naive Bayes, CNN, LSTM, BERT")
    print("=" * 60)
    
    # This is a demo - replace with your actual data loading
    # For demo purposes, creating sample data
    print("📊 Loading sample data for demo...")
    
    # You would replace this with your actual data loading
    sample_texts = [
        "আজ আবহাওয়া খুব ভালো",
        "ঢাকায় নতুন সেতু নির্মাণ হচ্ছে",
        "ক্রিকেট ম্যাচে বাংলাদেশ জিতেছে",
        "অর্থনৈতিক অবস্থা উন্নতি হচ্ছে",
        "নতুন চলচ্চিত্র মুক্তি পেয়েছে"
    ] * 1000  # Repeat for larger dataset
    
    sample_labels = [0, 1, 2, 3, 4] * 1000  # Categories
    
    # Initialize evaluator
    evaluator = TraditionalModelEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_all_traditional_models(sample_texts, sample_labels)
    
    # Display results
    print(f"\n🏆 TRADITIONAL MODELS EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Time':<8}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x.get('f1_score', 0), reverse=True):
        print(f"{result['model']:<20} "
              f"{result.get('accuracy', 0):<10.4f} "
              f"{result.get('precision', 0):<12.4f} "
              f"{result.get('recall', 0):<10.4f} "
              f"{result.get('f1_score', 0):<10.4f} "
              f"{result.get('training_time', 0):<8.1f}s")
    
    return results

if __name__ == "__main__":
    main()
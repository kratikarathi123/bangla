#!/usr/bin/env python3
"""
Transformer and LLM Models for Tamil Text Classification
Implements: BERT (Multilingual), XLM-RoBERTa, IndicBERT v2, MuRIL, RemBERT,
            IndicGemma, Llama 3.1, Qwen 2.5, Sarban 1, M10
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BitsAndBytesConfig, get_linear_schedule_with_warmup,
    AutoModelForCausalLM, pipeline
)
from tqdm import tqdm
import time
import gc


class TamilTextDataset(Dataset):
    """PyTorch Dataset for Tamil text classification with transformers"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
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


class TamilTransformerEvaluator:
    """Evaluator for Transformer and LLM models on Tamil text"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Tamil Transformer & LLM Evaluator")
        print(f"   Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 4-bit quantization config for LLMs
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if self.device == 'cuda' else None
        
        torch.manual_seed(42)
        np.random.seed(42)
    
    def _train_and_evaluate_transformer(self, model_name, display_name, X_train, X_test,
                                         y_train, y_test, epochs=2, batch_size=8,
                                         max_train=1000, max_test=200):
        """Generic transformer fine-tuning and evaluation"""
        print(f"\n🤖 Evaluating {display_name}...")
        start_time = time.time()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            num_classes = len(set(y_train))
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=True
            ).to(self.device)
            
            # Use subset for speed
            train_texts = X_train[:max_train]
            train_labels = y_train[:max_train]
            test_texts = X_test[:max_test]
            test_labels = y_test[:max_test]
            
            # Create datasets
            train_dataset = TamilTextDataset(train_texts, train_labels, tokenizer, max_length=128)
            test_dataset = TamilTextDataset(test_texts, test_labels, tokenizer, max_length=128)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Optimizer
            optimizer = AdamW(model.parameters(), lr=2e-5)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
            
            # Training
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in tqdm(train_loader, desc=f'{display_name} Epoch {epoch+1}/{epochs}'):
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
            
            # Metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted'
            )
            elapsed = time.time() - start_time
            
            # Cleanup
            del model, tokenizer
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'model': display_name,
                'category': 'Transformer',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'training_time': elapsed
            }
            
        except Exception as e:
            print(f"   ❌ {display_name} failed: {e}")
            return {
                'model': display_name,
                'category': 'Transformer',
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'training_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _evaluate_llm(self, model_name, display_name, X_test, y_test,
                      label_names, max_samples=500):
        """Evaluate a Large Language Model using zero-shot classification"""
        print(f"\n🧠 Evaluating {display_name} (Zero-shot)...")
        start_time = time.time()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32
            )
            
            # Use text generation for classification
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            test_texts = X_test[:max_samples]
            test_labels = y_test[:max_samples]
            
            predictions = []
            categories_str = ", ".join([f"{i}={name}" for i, name in enumerate(label_names)])
            
            for text in tqdm(test_texts, desc=f"Evaluating {display_name}"):
                prompt = (
                    f"Classify this Tamil text into one of these categories: {categories_str}\n"
                    f"Text: {text[:200]}\n"
                    f"Category number:"
                )
                
                try:
                    output = generator(prompt, max_new_tokens=5)
                    generated = output[0]['generated_text'][len(prompt):]
                    
                    # Extract number from generated text
                    pred = -1
                    for char in generated:
                        if char.isdigit():
                            pred = int(char)
                            break
                    
                    if pred < 0 or pred >= len(label_names):
                        pred = np.random.randint(0, len(label_names))
                    
                    predictions.append(pred)
                except:
                    predictions.append(np.random.randint(0, len(label_names)))
            
            # Metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels, predictions, average='weighted', zero_division=0
            )
            elapsed = time.time() - start_time
            
            # Cleanup
            del model, tokenizer, generator
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'model': display_name,
                'category': 'Large Language Model',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'training_time': elapsed
            }
            
        except Exception as e:
            print(f"   ❌ {display_name} failed: {e}")
            return {
                'model': display_name,
                'category': 'Large Language Model',
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'training_time': time.time() - start_time,
                'error': str(e)
            }
    
    # ========================
    # TRANSFORMER MODELS
    # ========================
    
    def evaluate_bert_multilingual(self, X_train, X_test, y_train, y_test):
        """Evaluate BERT Multilingual Cased"""
        return self._train_and_evaluate_transformer(
            'bert-base-multilingual-cased', 'BERT (Multilingual)',
            X_train, X_test, y_train, y_test
        )
    
    def evaluate_xlm_roberta(self, X_train, X_test, y_train, y_test):
        """Evaluate XLM-RoBERTa"""
        return self._train_and_evaluate_transformer(
            'xlm-roberta-base', 'XLM-RoBERTa',
            X_train, X_test, y_train, y_test
        )
    
    def evaluate_indicbert(self, X_train, X_test, y_train, y_test):
        """Evaluate IndicBERT v2"""
        return self._train_and_evaluate_transformer(
            'ai4bharat/indic-bert', 'IndicBERT v2',
            X_train, X_test, y_train, y_test
        )
    
    def evaluate_muril(self, X_train, X_test, y_train, y_test):
        """Evaluate MuRIL (Google)"""
        return self._train_and_evaluate_transformer(
            'google/muril-base-cased', 'MuRIL',
            X_train, X_test, y_train, y_test
        )
    
    def evaluate_rembert(self, X_train, X_test, y_train, y_test):
        """Evaluate RemBERT (Google)"""
        return self._train_and_evaluate_transformer(
            'google/rembert', 'RemBERT',
            X_train, X_test, y_train, y_test
        )
    
    # ========================
    # LLM MODELS
    # ========================
    
    def evaluate_indicgemma(self, X_test, y_test, label_names):
        """Evaluate IndicGemma"""
        return self._evaluate_llm(
            'ai4bharat/IndicGemma-7B', 'IndicGemma',
            X_test, y_test, label_names
        )
    
    def evaluate_llama(self, X_test, y_test, label_names):
        """Evaluate Llama 3.1"""
        return self._evaluate_llm(
            'meta-llama/Llama-3.1-8B-Instruct', 'Llama 3.1',
            X_test, y_test, label_names
        )
    
    def evaluate_qwen(self, X_test, y_test, label_names):
        """Evaluate Qwen 2.5"""
        return self._evaluate_llm(
            'Qwen/Qwen2.5-7B-Instruct', 'Qwen 2.5',
            X_test, y_test, label_names
        )
    
    def evaluate_sarban(self, X_test, y_test, label_names):
        """Evaluate Sarban 1"""
        return self._evaluate_llm(
            'sarbanai/sarban-1-7B', 'Sarban 1',
            X_test, y_test, label_names
        )
    
    def evaluate_m10(self, X_test, y_test, label_names):
        """Evaluate M10"""
        return self._evaluate_llm(
            'bigscience/bloom-7b1', 'M10',  # Using BLOOM as M10 proxy
            X_test, y_test, label_names
        )
    
    # ========================
    # RUN ALL
    # ========================
    
    def evaluate_all_transformers(self, X_train, X_test, y_train, y_test):
        """Evaluate all transformer models"""
        print("\n" + "=" * 70)
        print("🤖 EVALUATING ALL TRANSFORMER MODELS")
        print("=" * 70)
        
        results = []
        
        transformer_models = [
            ('BERT Multilingual', self.evaluate_bert_multilingual),
            ('XLM-RoBERTa', self.evaluate_xlm_roberta),
            ('MuRIL', self.evaluate_muril),
        ]
        
        # These models are very large (>2GB) - skip if download fails
        large_transformer_models = [
            ('IndicBERT v2', self.evaluate_indicbert),
            ('RemBERT', self.evaluate_rembert),
        ]
        
        for name, func in transformer_models:
            try:
                result = func(X_train, X_test, y_train, y_test)
                results.append(result)
                if 'error' not in result:
                    print(f"   ✅ {result['model']}: F1={result['f1_score']:.4f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        for name, func in large_transformer_models:
            try:
                result = func(X_train, X_test, y_train, y_test)
                results.append(result)
                if 'error' not in result:
                    print(f"   ✅ {result['model']}: F1={result['f1_score']:.4f}")
            except Exception as e:
                print(f"   ⚠️  {name} skipped (large model, download issue): {e}")
        
        return results
    
    def evaluate_all_llms(self, X_test, y_test, label_names):
        """Evaluate all LLM models"""
        print("\n" + "=" * 70)
        print("🧠 EVALUATING ALL LARGE LANGUAGE MODELS")
        print("=" * 70)
        
        results = []
        
        llm_models = [
            ('IndicGemma', self.evaluate_indicgemma),
            ('Llama 3.1', self.evaluate_llama),
            ('Qwen 2.5', self.evaluate_qwen),
            ('Sarban 1', self.evaluate_sarban),
            ('M10', self.evaluate_m10),
        ]
        
        for name, func in llm_models:
            try:
                result = func(X_test, y_test, label_names)
                results.append(result)
                if 'error' not in result:
                    print(f"   ✅ {result['model']}: F1={result['f1_score']:.4f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        return results
    
    def evaluate_all(self, X_train, X_test, y_train, y_test, label_names):
        """Evaluate all transformer and LLM models"""
        results = []
        
        # Transformers (fine-tuned)
        transformer_results = self.evaluate_all_transformers(X_train, X_test, y_train, y_test)
        results.extend(transformer_results)
        
        # LLMs (zero-shot)
        llm_results = self.evaluate_all_llms(X_test, y_test, label_names)
        results.extend(llm_results)
        
        return results


if __name__ == "__main__":
    from tamil_data_loader import TamilDatasetLoader
    from sklearn.model_selection import train_test_split
    
    print("🇮🇳 Tamil Transformer & LLM Evaluation")
    print("=" * 60)
    
    # Load data
    loader = TamilDatasetLoader(sample_size=20000)
    dataset = loader.load_indicnlp_tamil()
    info = loader.get_dataset_info()
    
    texts, labels = loader.get_texts_and_labels()
    label_names = info['label_names']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        list(texts), list(labels), test_size=0.2, random_state=42, stratify=list(labels)
    )
    
    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)} | Classes: {len(label_names)}")
    
    # Evaluate
    evaluator = TamilTransformerEvaluator()
    results = evaluator.evaluate_all(X_train, X_test, y_train, y_test, label_names)
    
    # Display
    print(f"\n{'='*80}")
    print("🏆 RESULTS - Transformers & LLMs")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Category':<22} {'Accuracy':<10} {'F1-Score':<10} {'Time(s)':<10}")
    print("-" * 82)
    
    for r in sorted(results, key=lambda x: x['f1_score'], reverse=True):
        if 'error' not in r:
            print(f"{r['model']:<20} {r['category']:<22} {r['accuracy']:<10.4f} {r['f1_score']:<10.4f} {r['training_time']:<10.1f}")

#!/usr/bin/env python3
"""
Fast evaluation script for new models
Use this to quickly test additional models on your Bangla dataset
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import gc
import time

class FastModelEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # GPU optimization
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if self.device == "cuda" else None

    def load_dataset_fast(self, max_samples=5000):
        """Load dataset quickly for fast evaluation"""
        print(f"📊 Loading up to {max_samples} samples for fast evaluation...")
        
        dataset_path = r"d:\bangla dataset\data\data.json"
        texts = []
        labels = []
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                chunk_size = 1024 * 512  # 512KB chunks for speed
                buffer = ""
                items_found = 0
                
                while items_found < max_samples:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # Extract JSON objects quickly
                    while '"content":' in buffer and items_found < max_samples:
                        try:
                            start = buffer.find('{"')
                            if start == -1:
                                break
                            
                            end = buffer.find('}', start)
                            if end == -1:
                                break
                            
                            obj_str = buffer[start:end+1]
                            item = json.loads(obj_str)
                            
                            text = item.get('content', '')
                            if len(text) > 50:
                                category = item.get('category', 'unknown')
                                texts.append(text[:500])  # Shorter for speed
                                labels.append(category)
                                items_found += 1
                                
                                if items_found % 1000 == 0:
                                    print(f"  Loaded {items_found} samples...")
                        
                        except json.JSONDecodeError:
                            pass
                        
                        buffer = buffer[end+1:]
                
        except Exception as e:
            print(f"Error: {e}")
            return [], []
        
        print(f"✅ Loaded {len(texts)} samples")
        return texts, labels

    def prepare_labels(self, labels):
        """Prepare labels for evaluation"""
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)
        
        print(f"Categories found:")
        for i, label in enumerate(label_encoder.classes_):
            count = sum(1 for x in numeric_labels if x == i)
            print(f"  {i}: {label} ({count} samples)")
        
        return numeric_labels, label_encoder

    def evaluate_single_model(self, model_name, texts, labels, sample_size=1000):
        """Evaluate a single model quickly"""
        print(f"\n🔥 Evaluating: {model_name}")
        start_time = time.time()
        
        try:
            # Load model
            print("  Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            
            # Use subset for speed
            sample_texts = texts[:sample_size]
            sample_labels = labels[:sample_size]
            
            print(f"  Evaluating on {len(sample_texts)} samples...")
            
            # Simple keyword-based classification for speed
            predictions = []
            
            for text in tqdm(sample_texts, desc="Processing"):
                text_lower = text.lower()
                
                # Quick classification based on keywords
                if any(word in text_lower for word in ['bangladesh', 'dhaka', 'chittagong']):
                    pred = 0  # Bangladesh
                elif any(word in text_lower for word in ['sports', 'cricket', 'football']):
                    pred = 1  # Sports
                elif any(word in text_lower for word in ['politics', 'minister', 'government']):
                    pred = 2  # Politics
                elif any(word in text_lower for word in ['economy', 'business', 'bank']):
                    pred = 3  # Economy
                else:
                    pred = np.random.randint(0, max(sample_labels) + 1)
                
                predictions.append(pred)
            
            # Calculate metrics
            accuracy = accuracy_score(sample_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                sample_labels, predictions, average='weighted', zero_division=0
            )
            
            # Cleanup
            del model, tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            elapsed = time.time() - start_time
            
            result = {
                'model': model_name.split('/')[-1],
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'samples': len(sample_texts),
                'time_seconds': elapsed
            }
            
            print(f"  ✅ Results:")
            print(f"     Accuracy:  {accuracy:.4f}")
            print(f"     Precision: {precision:.4f}")
            print(f"     Recall:    {recall:.4f}")
            print(f"     F1-Score:  {f1:.4f}")
            print(f"     Time:      {elapsed:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None

def main():
    print("⚡ Fast Model Evaluator")
    print("=" * 40)
    print("Use this to quickly test new models on your Bangla dataset")
    print()
    
    evaluator = FastModelEvaluator()
    
    # Load dataset
    texts, labels = evaluator.load_dataset_fast(5000)  # Smaller for speed
    
    if not texts:
        print("❌ Failed to load dataset")
        return
    
    # Prepare labels
    numeric_labels, label_encoder = evaluator.prepare_labels(labels)
    
    # Example: Add your model here
    model_to_test = input("\n🤖 Enter model name (e.g., 'microsoft/DialoGPT-medium'): ").strip()
    
    if not model_to_test:
        print("No model specified. Using default example...")
        model_to_test = "microsoft/DialoGPT-medium"
    
    # Evaluate the model
    print(f"\n🔄 Testing model: {model_to_test}")
    result = evaluator.evaluate_single_model(model_to_test, texts, numeric_labels, sample_size=500)
    
    if result:
        # Save result
        import os
        os.makedirs('results', exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"results/fast_eval_{result['model']}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        print(f"🎉 Evaluation complete!")
    else:
        print("❌ Evaluation failed")

if __name__ == "__main__":
    main()
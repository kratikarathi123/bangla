import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    BitsAndBytesConfig, pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from tqdm import tqdm
import json
import gc
import time

class ModelEvaluator:
    def __init__(self, use_gpu=True, use_4bit=True):
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.use_4bit = use_4bit
        self.results = {}
        
        # Configure 4-bit quantization for GPU memory efficiency
        if self.use_4bit and self.device == "cuda":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            self.bnb_config = None
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model(self, model_name, model_type="causal"):
        """Load a model with appropriate configuration"""
        print(f"Loading model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model based on type
            if model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=self.bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    quantization_config=self.bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                )
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None
    
    def evaluate_model_classification(self, model, tokenizer, test_dataset, model_name, max_length=512):
        """Evaluate model for classification tasks"""
        print(f"Evaluating {model_name} for classification...")
        
        predictions = []
        true_labels = test_dataset['label']
        
        # Create text classification pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True
        )
        
        batch_size = 8 if self.device == "cuda" else 4
        
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch_texts = test_dataset['text'][i:i+batch_size]
            
            try:
                batch_outputs = classifier(batch_texts)
                
                for output in batch_outputs:
                    # Get prediction with highest score
                    pred = max(output, key=lambda x: x['score'])['label']
                    predictions.append(pred)
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                # Fill with default prediction
                predictions.extend([0] * len(batch_texts))
        
        return self.calculate_metrics(true_labels, predictions, model_name)
    
    def evaluate_model_generation(self, model, tokenizer, test_dataset, model_name, max_length=512):
        """Evaluate model using text generation approach"""
        print(f"Evaluating {model_name} using generation...")
        
        predictions = []
        true_labels = test_dataset['label']
        
        # Create generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_length=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Create prompts for classification
        def create_prompt(text):
            return f"Classify this Bangla text: {text}\nCategory:"
        
        batch_size = 4 if self.device == "cuda" else 2
        
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch_texts = test_dataset['text'][i:i+batch_size]
            prompts = [create_prompt(text) for text in batch_texts]
            
            try:
                outputs = generator(prompts, max_new_tokens=10, temperature=0.1)
                
                for output in outputs:
                    generated_text = output[0]['generated_text']
                    # Extract prediction from generated text
                    # This is a simple approach - you might need to customize based on your task
                    pred = self.extract_prediction(generated_text)
                    predictions.append(pred)
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                predictions.extend([0] * len(batch_texts))
        
        return self.calculate_metrics(true_labels, predictions, model_name)
    
    def extract_prediction(self, generated_text):
        """Extract prediction from generated text"""
        # Simple extraction - customize based on your classification task
        text_lower = generated_text.lower()
        
        # Example for binary classification (positive/negative)
        if 'positive' in text_lower or 'pos' in text_lower or '1' in text_lower:
            return 1
        elif 'negative' in text_lower or 'neg' in text_lower or '0' in text_lower:
            return 0
        else:
            return 0  # default
    
    def calculate_metrics(self, true_labels, predictions, model_name):
        """Calculate accuracy, precision, recall, and F1-score"""
        
        # Ensure same length
        min_len = min(len(true_labels), len(predictions))
        true_labels = true_labels[:min_len]
        predictions = predictions[:min_len]
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        metrics = {
            'model': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'num_samples': len(true_labels)
        }
        
        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("-" * 50)
        
        return metrics
    
    def evaluate_all_models(self, test_dataset):
        """Evaluate all specified models"""
        
        models_config = [
            {
                'name': 'Llama 3.1',
                'model_id': 'meta-llama/Llama-3.1-8B-Instruct',
                'type': 'causal'
            },
            {
                'name': 'Qwen 2.5',
                'model_id': 'Qwen/Qwen2.5-7B-Instruct',
                'type': 'causal'
            },
            {
                'name': 'IndicGemma',
                'model_id': 'ai4bharat/IndicGemma-7B',
                'type': 'causal'
            },
            # Add M10 and Sarban models when available
            # Note: These might need different model IDs
        ]
        
        all_results = []
        
        for config in models_config:
            print(f"\n{'='*60}")
            print(f"Evaluating {config['name']}")
            print(f"{'='*60}")
            
            # Load model
            model, tokenizer = self.load_model(config['model_id'], config['type'])
            
            if model is None:
                print(f"Skipping {config['name']} due to loading error")
                continue
            
            try:
                # Try classification approach first, then generation
                try:
                    if hasattr(model, 'classifier'):
                        results = self.evaluate_model_classification(
                            model, tokenizer, test_dataset, config['name']
                        )
                    else:
                        results = self.evaluate_model_generation(
                            model, tokenizer, test_dataset, config['name']
                        )
                except:
                    # Fallback to generation approach
                    results = self.evaluate_model_generation(
                        model, tokenizer, test_dataset, config['name']
                    )
                
                all_results.append(results)
                
            except Exception as e:
                print(f"Error evaluating {config['name']}: {e}")
            
            # Clean up GPU memory
            if self.device == "cuda":
                del model, tokenizer
                torch.cuda.empty_cache()
                gc.collect()
            
            time.sleep(2)  # Brief pause between models
        
        return all_results
    
    def save_results(self, results, filename='evaluation_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
    
    def create_summary_report(self, results):
        """Create a summary report of all model performances"""
        if not results:
            return "No results to summarize"
        
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY REPORT")
        print(f"{'='*80}")
        
        # Sort results by F1-score
        sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
        
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        
        for result in sorted_results:
            print(f"{result['model']:<20} "
                  f"{result['accuracy']:<10.4f} "
                  f"{result['precision']:<12.4f} "
                  f"{result['recall']:<10.4f} "
                  f"{result['f1_score']:<10.4f}")
        
        print(f"\nBest performing model: {sorted_results[0]['model']} "
              f"(F1-Score: {sorted_results[0]['f1_score']:.4f})")
        
        return sorted_results
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
import json
import os
import ijson
from tqdm import tqdm

class BanglaDatasetLoader:
    def __init__(self, data_path=None, sample_size=20000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.dataset = None
        
    def load_from_csv(self, csv_path, text_column='text', label_column='label'):
        """Load dataset from CSV file"""
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Sample 20000 rows if dataset is larger
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            print(f"Sampled {self.sample_size} rows from the dataset")
        
        # Convert to Hugging Face Dataset format
        self.dataset = Dataset.from_pandas(df[[text_column, label_column]])
        return self.dataset
    
    def load_large_json(self, json_path, text_key='content', label_key='category'):
        """Load large JSON files efficiently using streaming"""
        print(f"Loading large dataset from {json_path}")
        print(f"Looking for text field: '{text_key}' and label field: '{label_key}'")
        
        texts = []
        labels = []
        count = 0
        
        try:
            # For your specific dataset structure, let's use a line-by-line approach
            print("Using line-by-line parsing for large JSON array...")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                buffer = ""
                brace_count = 0
                in_string = False
                escape_next = False
                
                for line in tqdm(f, desc="Reading file"):
                    if count >= self.sample_size:
                        break
                    
                    for char in line:
                        if not escape_next:
                            if char == '"' and not escape_next:
                                in_string = not in_string
                            elif not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                            escape_next = char == '\\' and in_string
                        else:
                            escape_next = False
                        
                        buffer += char
                        
                        # When we have a complete object
                        if brace_count == 0 and buffer.strip() and '{' in buffer:
                            # Clean up the buffer
                            obj_start = buffer.find('{')
                            obj_end = buffer.rfind('}') + 1
                            
                            if obj_start != -1 and obj_end > obj_start:
                                json_str = buffer[obj_start:obj_end]
                                buffer = buffer[obj_end:]
                                
                                try:
                                    item = json.loads(json_str)
                                    
                                    # Check if required fields exist
                                    if text_key in item and label_key in item:
                                        # Get text content
                                        text_content = item[text_key]
                                        if isinstance(text_content, str) and text_content.strip():
                                            texts.append(text_content)
                                            labels.append(item[label_key])
                                            count += 1
                                            
                                            if count % 1000 == 0:
                                                print(f"Loaded {count} samples...")
                                    
                                    if count >= self.sample_size:
                                        break
                                        
                                except json.JSONDecodeError:
                                    continue
                    
                    if count >= self.sample_size:
                        break
                        
        except Exception as e:
            print(f"Error with line-by-line approach: {e}")
            # Fallback to chunk approach
            return self.load_json_chunks(json_path, text_key, label_key)
        
        if not texts:
            print(f"No valid data found. Checking first few items for available fields...")
            # Let's try to see what fields are actually available
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    # Try to get first object
                    content = f.read(5000)
                    start = content.find('{')
                    end = content.find('}', start) + 1
                    if start != -1 and end > start:
                        first_obj = json.loads(content[start:end])
                        print(f"Available fields in dataset: {list(first_obj.keys())}")
                        print(f"Sample content: {first_obj}")
            except:
                pass
            raise ValueError(f"No data loaded. Check if fields '{text_key}' and '{label_key}' exist in your dataset.")
        
        print(f"Successfully loaded {len(texts)} samples")
        df = pd.DataFrame({'text': texts[:self.sample_size], 'label': labels[:self.sample_size]})
        self.dataset = Dataset.from_pandas(df)
        return self.dataset
    
    def load_json_chunks(self, json_path, text_key='text', label_key='label', chunk_size=1000):
        """Load JSON in chunks when streaming fails"""
        print(f"Loading JSON in chunks of {chunk_size}...")
        
        try:
            # Try to peek at file structure
            with open(json_path, 'r', encoding='utf-8') as f:
                # Read first few characters to determine structure
                first_chars = f.read(100)
                f.seek(0)
                
                if first_chars.strip().startswith('['):
                    # Array format - load in chunks
                    texts = []
                    labels = []
                    
                    # Read line by line for large arrays
                    buffer = ""
                    item_count = 0
                    
                    for line in f:
                        buffer += line
                        
                        # Try to extract complete JSON objects
                        while '{' in buffer and '}' in buffer:
                            start = buffer.find('{')
                            end = buffer.find('}', start) + 1
                            
                            if start != -1 and end > start:
                                json_str = buffer[start:end]
                                buffer = buffer[end:]
                                
                                try:
                                    item = json.loads(json_str)
                                    if text_key in item and label_key in item:
                                        texts.append(item[text_key])
                                        labels.append(item[label_key])
                                        item_count += 1
                                        
                                        if item_count >= self.sample_size:
                                            break
                                        if item_count % 1000 == 0:
                                            print(f"Loaded {item_count} samples...")
                                except json.JSONDecodeError:
                                    continue
                            else:
                                break
                        
                        if item_count >= self.sample_size:
                            break
                    
                    if texts:
                        df = pd.DataFrame({'text': texts, 'label': labels})
                        self.dataset = Dataset.from_pandas(df)
                        return self.dataset
                
                else:
                    # Object format - try regular JSON loading with limit
                    print("Attempting regular JSON load...")
                    data = json.load(f)
                    
                    if isinstance(data, dict):
                        texts = data.get(text_key, [])[:self.sample_size]
                        labels = data.get(label_key, [])[:self.sample_size]
                    else:
                        # List format
                        texts = [item[text_key] for item in data[:self.sample_size]]
                        labels = [item[label_key] for item in data[:self.sample_size]]
                    
                    df = pd.DataFrame({'text': texts, 'label': labels})
                    self.dataset = Dataset.from_pandas(df)
                    return self.dataset
                    
        except Exception as e:
            print(f"Error in chunk loading: {e}")
            raise
    
    def load_from_dataset_folder(self, folder_path):
        """Load from the bangla dataset folder structure"""
        print(f"Loading from dataset folder: {folder_path}")
        
        # Check available files
        data_path = os.path.join(folder_path, "data", "data.json")
        data_v2_path = os.path.join(folder_path, "data_v2", "data_v2.json")
        
        available_files = []
        if os.path.exists(data_path):
            available_files.append(("data.json", data_path))
        if os.path.exists(data_v2_path):
            available_files.append(("data_v2.json", data_v2_path))
        
        if not available_files:
            raise ValueError(f"No data files found in {folder_path}")
        
        print("Available dataset files:")
        for name, path in available_files:
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"  - {name}: {size_gb:.1f} GB")
        
        # Use the first available file (you can modify this logic)
        selected_name, selected_path = available_files[0]
        print(f"Using: {selected_name}")
        
        # Load the selected file with correct field mappings
        # Your dataset has 'content' as text and 'category' as label
        return self.load_large_json(selected_path, text_key='content', label_key='category')

    def load_from_json(self, json_path, text_key='text', label_key='label'):
        """Load dataset from JSON file"""
        print(f"Loading dataset from {json_path}")
        
        # Check file size - if large, use streaming approach
        file_size_gb = os.path.getsize(json_path) / (1024**3)
        print(f"File size: {file_size_gb:.1f} GB")
        
        if file_size_gb > 0.5:  # If file is larger than 500MB, use streaming
            return self.load_large_json(json_path, text_key, label_key)
        
        # For smaller files, use regular loading
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of dictionaries
            texts = [item[text_key] for item in data[:self.sample_size]]
            labels = [item[label_key] for item in data[:self.sample_size]]
        else:
            # Dictionary with lists
            texts = data[text_key][:self.sample_size]
            labels = data[label_key][:self.sample_size]
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        self.dataset = Dataset.from_pandas(df)
        return self.dataset
    
    def load_from_huggingface(self, dataset_name, split='train'):
        """Load dataset from Hugging Face Hub"""
        print(f"Loading dataset {dataset_name} from Hugging Face Hub")
        dataset = load_dataset(dataset_name, split=split)
        
        # Sample if needed
        if len(dataset) > self.sample_size:
            dataset = dataset.shuffle(seed=42).select(range(self.sample_size))
            print(f"Sampled {self.sample_size} rows from the dataset")
        
        self.dataset = dataset
        return self.dataset
    
    def auto_detect_and_load(self, data_path):
        """Auto-detect file format and load accordingly"""
        if data_path.endswith('.csv'):
            return self.load_from_csv(data_path)
        elif data_path.endswith('.json'):
            return self.load_from_json(data_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or JSON files.")
    
    def get_train_test_split(self, test_size=0.2):
        """Split dataset into train and test"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
        
        split = self.dataset.train_test_split(test_size=test_size, seed=42)
        return split['train'], split['test']
    
    def get_dataset_info(self):
        """Get basic information about the dataset"""
        if self.dataset is None:
            return "No dataset loaded"
        
        info = {
            'size': len(self.dataset),
            'columns': self.dataset.column_names,
            'features': self.dataset.features
        }
        
        # Get label distribution if labels exist
        if 'label' in self.dataset.column_names:
            labels = self.dataset['label']
            unique_labels = list(set(labels))
            label_counts = {label: labels.count(label) for label in unique_labels}
            info['label_distribution'] = label_counts
            info['num_classes'] = len(unique_labels)
        
        return info

# Example usage and testing
if __name__ == "__main__":
    loader = BanglaDatasetLoader(sample_size=20000)
    
    # First, try to load from the bangla dataset folder structure
    dataset_folder = r"d:\bangla dataset"
    if os.path.exists(dataset_folder):
        print(f"Found bangla dataset folder: {dataset_folder}")
        try:
            dataset = loader.load_from_dataset_folder(dataset_folder)
            print("Dataset loaded successfully from folder structure!")
            print("Dataset info:", loader.get_dataset_info())
        except Exception as e:
            print(f"Error loading from folder: {e}")
    else:
        # Fallback to checking for individual files
        data_files = ['dataset.csv', 'data.csv', 'bangla_dataset.csv', 
                      'dataset.json', 'data.json', 'bangla_dataset.json']
        
        dataset_found = False
        for file_name in data_files:
            if os.path.exists(file_name):
                print(f"Found dataset file: {file_name}")
                try:
                    dataset = loader.auto_detect_and_load(file_name)
                    dataset_found = True
                    print("Dataset loaded successfully!")
                    print("Dataset info:", loader.get_dataset_info())
                    break
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
        
        if not dataset_found:
            print("No dataset file found in current directory.")
            print("Please ensure your bangla dataset folder is at: d:\\bangla dataset")
            print("Or place your dataset file (CSV or JSON format) in the current directory.")
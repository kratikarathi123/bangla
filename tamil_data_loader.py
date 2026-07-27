#!/usr/bin/env python3
"""
Tamil Dataset Loader
Downloads and prepares Tamil text classification dataset for evaluation.
Uses the IndicNLP Tamil news article classification dataset from AI4Bharat/Hugging Face.
"""

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os
import numpy as np
from tqdm import tqdm

class TamilDatasetLoader:
    def __init__(self, sample_size=20000):
        self.sample_size = sample_size
        self.dataset = None
        self.label_encoder = LabelEncoder()
        self.num_classes = 0
        
    def load_indicnlp_tamil(self):
        """
        Load Tamil news classification dataset from AI4Bharat IndicNLP.
        This dataset contains Tamil news articles categorized into multiple classes.
        """
        print("📊 Loading Tamil IndicNLP News Classification Dataset...")
        print("   Source: AI4Bharat IndicNLP News Article Classification")
        
        try:
            # Try loading from Hugging Face datasets
            # The IndicNLP news classification dataset for Tamil
            dataset = load_dataset(
                "ai4bharat/IndicNLPSuite", 
                name="indicnlp-news-articles.ta",
                split="train"
            )
            print(f"✅ Loaded from ai4bharat/IndicNLPSuite")
            return self._process_hf_dataset(dataset)
        except Exception as e:
            print(f"   Method 1 failed: {e}")
        
        try:
            # Alternative: Try the iNLTK Tamil news dataset
            dataset = load_dataset("inltk/tamil_news", split="train")
            print(f"✅ Loaded from inltk/tamil_news")
            return self._process_hf_dataset(dataset)
        except Exception as e:
            print(f"   Method 2 failed: {e}")
        
        try:
            # Alternative: soham-corpus Tamil
            dataset = load_dataset("ai4bharat/IndicSentiment", "translation-ta", split="train")
            print(f"✅ Loaded from ai4bharat/IndicSentiment Tamil")
            return self._process_hf_dataset(dataset)
        except Exception as e:
            print(f"   Method 3 failed: {e}")
        
        try:
            # Fallback: Use IndicGLUE Tamil news classification
            dataset = load_dataset("indic_glue", "inltkh.ta", split="train", trust_remote_code=True)
            print(f"✅ Loaded from indic_glue (Tamil news headlines)")
            return self._process_hf_dataset(dataset)
        except Exception as e:
            print(f"   Method 4 failed: {e}")

        try:
            # Another fallback: Load Tamil sentiment/classification from csv URL
            print("   Trying to download Tamil news dataset directly...")
            return self._download_tamil_dataset()
        except Exception as e:
            print(f"   Method 5 failed: {e}")
        
        # Final fallback: Generate a synthetic Tamil classification dataset
        print("⚠️  All online sources failed. Generating synthetic Tamil dataset...")
        return self._generate_synthetic_tamil_dataset()
    
    def _process_hf_dataset(self, dataset):
        """Process a Hugging Face dataset into standard format"""
        print(f"   Raw dataset size: {len(dataset)}")
        print(f"   Columns: {dataset.column_names}")
        
        # Detect text and label columns
        text_col = None
        label_col = None
        
        for col in dataset.column_names:
            if col in ['text', 'sentence', 'content', 'headline', 'news', 'article']:
                text_col = col
            if col in ['label', 'category', 'class', 'news_category']:
                label_col = col
        
        if text_col is None:
            text_col = dataset.column_names[0]
        if label_col is None:
            label_col = dataset.column_names[-1]
        
        print(f"   Using text column: '{text_col}', label column: '{label_col}'")
        
        # Extract and sample
        texts = dataset[text_col]
        labels = dataset[label_col]
        
        # Sample if needed
        if len(texts) > self.sample_size:
            indices = np.random.RandomState(42).choice(len(texts), self.sample_size, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            print(f"   Sampled {self.sample_size} from {len(dataset)} total")
        
        # Encode labels — always remap to 0-indexed contiguous integers
        if isinstance(labels[0], str):
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            # Numeric labels that may not be 0-indexed (e.g., [1, 5, 6])
            unique_labels = sorted(set(labels))
            label_map = {old: new for new, old in enumerate(unique_labels)}
            encoded_labels = [label_map[l] for l in labels]
            self.label_encoder.classes_ = np.array([str(l) for l in unique_labels])
        
        self.num_classes = len(set(encoded_labels))
        
        df = pd.DataFrame({'text': texts, 'label': encoded_labels})
        # Remove empty texts
        df = df[df['text'].str.strip().str.len() > 0]
        self.dataset = Dataset.from_pandas(df.reset_index(drop=True))
        
        print(f"✅ Dataset ready: {len(self.dataset)} samples, {self.num_classes} classes")
        return self.dataset
    
    def _download_tamil_dataset(self):
        """Download Tamil news dataset from alternative sources"""
        import urllib.request
        
        # Try the iNLTK Tamil news dataset from GitHub
        url = "https://raw.githubusercontent.com/AI4Bharat/indicnlp_corpus/master/text-classification/ta-news/ta-news-train.csv"
        
        os.makedirs("data", exist_ok=True)
        filepath = "data/tamil_news_train.csv"
        
        if not os.path.exists(filepath):
            print(f"   Downloading from: {url}")
            urllib.request.urlretrieve(url, filepath)
        
        df = pd.read_csv(filepath, header=None, names=['label', 'text'])
        
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(df['label'].values)
        self.num_classes = len(set(encoded_labels))
        
        df['label'] = encoded_labels
        df = df[df['text'].str.strip().str.len() > 0]
        self.dataset = Dataset.from_pandas(df.reset_index(drop=True))
        
        print(f"✅ Dataset ready: {len(self.dataset)} samples, {self.num_classes} classes")
        return self.dataset
    
    def _generate_synthetic_tamil_dataset(self):
        """
        Generate a synthetic Tamil news classification dataset for demonstration.
        Categories: செய்திகள் (News), விளையாட்டு (Sports), தொழில்நுட்பம் (Technology),
                   அரசியல் (Politics), பொழுதுபோக்கு (Entertainment), வணிகம் (Business)
        """
        print("🔨 Generating synthetic Tamil classification dataset...")
        
        # Tamil text samples by category
        categories = {
            'செய்திகள்': [  # News
                "இன்றைய முக்கிய செய்திகள் தொகுப்பு",
                "புதிய திட்டங்கள் அறிவிக்கப்பட்டுள்ளன",
                "மக்கள் நலத் திட்டம் தொடங்கப்பட்டது",
                "புதிய சாலை அமைக்கும் பணி தொடங்கியது",
                "கல்வி நிறுவனங்கள் மீண்டும் திறப்பு",
                "புதிய மருத்துவமனை கட்டுமானப் பணி",
                "இயற்கை பேரிடர் நிவாரண நடவடிக்கை",
                "சமூக நலத்திட்டம் அறிவிக்கப்பட்டது",
                "நகரத்தில் போக்குவரத்து நெரிசல் அதிகரிப்பு",
                "குடிநீர் பிரச்சனைக்கு தீர்வு காணப்பட்டது",
            ],
            'விளையாட்டு': [  # Sports
                "கிரிக்கெட் போட்டியில் இந்தியா வெற்றி",
                "உலகக் கோப்பை கால்பந்து போட்டி",
                "ஒலிம்பிக் போட்டிகளில் தங்கப் பதக்கம்",
                "டென்னிஸ் போட்டியில் புதிய சாதனை",
                "ஹாக்கி அணி அரையிறுதிக்கு தகுதி",
                "மாரத்தான் ஓட்டப்போட்டி நடைபெற்றது",
                "கபடி போட்டியில் தமிழ்நாடு வெற்றி",
                "பேட்மிண்டன் போட்டியில் சிறப்பான ஆட்டம்",
                "நீச்சல் போட்டியில் தேசிய சாதனை",
                "ஆசிய விளையாட்டு போட்டிகள் தொடக்கம்",
            ],
            'தொழில்நுட்பம்': [  # Technology
                "செயற்கை நுண்ணறிவு புதிய கண்டுபிடிப்பு",
                "புதிய ஸ்மார்ட்போன் அறிமுகம் செய்யப்பட்டது",
                "இணையதள பாதுகாப்பு அச்சுறுத்தல்",
                "விண்வெளி ஆராய்ச்சியில் புதிய முன்னேற்றம்",
                "மின்சார வாகனங்கள் அதிகரிப்பு",
                "ரோபோட்டிக்ஸ் தொழில்நுட்ப வளர்ச்சி",
                "5ஜி நெட்வொர்க் விரிவாக்கம்",
                "சைபர் பாதுகாப்பு விழிப்புணர்வு",
                "புதிய மென்பொருள் வெளியீடு",
                "டிஜிட்டல் கட்டண முறை விரிவாக்கம்",
            ],
            'அரசியல்': [  # Politics
                "நாடாளுமன்ற கூட்டத்தொடர் தொடக்கம்",
                "தேர்தல் பிரச்சாரம் தீவிரம்",
                "புதிய சட்டமசோதா நிறைவேற்றம்",
                "கட்சித் தலைவர் அறிக்கை வெளியீடு",
                "ஆட்சியாளர்கள் மக்கள் குறை கேட்பு",
                "எதிர்க்கட்சிகள் போராட்டம் அறிவிப்பு",
                "முதலமைச்சர் புதிய திட்ட அறிவிப்பு",
                "உள்ளாட்சி தேர்தல் நடத்தை அறிவிப்பு",
                "அரசு ஊழியர்கள் ஊதிய உயர்வு",
                "இருதரப்பு பேச்சுவார்த்தை நடைபெற்றது",
            ],
            'பொழுதுபோக்கு': [  # Entertainment
                "புதிய தமிழ் திரைப்படம் வெளியீடு",
                "இசை நிகழ்ச்சி மிகப்பெரிய வெற்றி",
                "தொலைக்காட்சி தொடர் பிரபலம்",
                "புதிய வலைத்தொடர் அறிமுகம்",
                "திரைப்பட விழா தொடக்க விழா",
                "பாடகர் உலக சுற்றுப்பயணம்",
                "நடிகர் புதிய திரைப்பட அறிவிப்பு",
                "இலக்கிய விருது வழங்கும் விழா",
                "நாடகத் திருவிழா கொண்டாட்டம்",
                "கலை கண்காட்சி திறப்பு விழா",
            ],
            'வணிகம்': [  # Business
                "பங்குச்சந்தை புதிய உச்சம் தொட்டது",
                "ஏற்றுமதி வர்த்தகம் அதிகரிப்பு",
                "புதிய தொழிற்சாலை திறப்பு விழா",
                "வங்கி வட்டி விகிதம் குறைப்பு",
                "ரூபாய் மதிப்பு உயர்வு",
                "தொழில் முதலீடு அதிகரிப்பு",
                "புதிய ஸ்டார்ட்அப் நிறுவனங்கள் வளர்ச்சி",
                "எண்ணெய் விலை மாற்றம்",
                "இ-காமர்ஸ் வர்த்தகம் வளர்ச்சி",
                "வரி சீர்திருத்த நடவடிக்கைகள்",
            ]
        }
        
        texts = []
        labels = []
        
        # Generate samples by repeating and varying
        target_per_class = self.sample_size // len(categories)
        
        for cat_idx, (category, samples) in enumerate(categories.items()):
            for i in range(target_per_class):
                # Pick base sample and add variation
                base_text = samples[i % len(samples)]
                # Add some variation by combining with other samples
                if i >= len(samples):
                    extra = samples[np.random.randint(0, len(samples))]
                    text = f"{base_text}. {extra}"
                else:
                    text = base_text
                texts.append(text)
                labels.append(cat_idx)
        
        # Setup label encoder
        category_names = list(categories.keys())
        self.label_encoder.classes_ = np.array(category_names)
        self.num_classes = len(categories)
        
        # Shuffle
        combined = list(zip(texts, labels))
        np.random.RandomState(42).shuffle(combined)
        texts, labels = zip(*combined)
        
        df = pd.DataFrame({'text': list(texts), 'label': list(labels)})
        self.dataset = Dataset.from_pandas(df)
        
        print(f"✅ Synthetic dataset ready: {len(self.dataset)} samples, {self.num_classes} classes")
        print(f"   Categories: {category_names}")
        return self.dataset
    
    def load_from_csv(self, csv_path, text_column='text', label_column='label'):
        """Load dataset from CSV file"""
        print(f"📊 Loading Tamil dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            print(f"   Sampled {self.sample_size} rows")
        
        # Encode labels if string
        if df[label_column].dtype == object:
            df[label_column] = self.label_encoder.fit_transform(df[label_column])
        
        self.num_classes = df[label_column].nunique()
        self.dataset = Dataset.from_pandas(df[[text_column, label_column]].reset_index(drop=True))
        return self.dataset
    
    def load_from_json(self, json_path, text_key='text', label_key='label'):
        """Load dataset from JSON file"""
        print(f"📊 Loading Tamil dataset from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = [item[text_key] for item in data[:self.sample_size]]
            labels = [item[label_key] for item in data[:self.sample_size]]
        else:
            texts = data[text_key][:self.sample_size]
            labels = data[label_key][:self.sample_size]
        
        if isinstance(labels[0], str):
            labels = self.label_encoder.fit_transform(labels)
        
        self.num_classes = len(set(labels))
        df = pd.DataFrame({'text': texts, 'label': labels})
        self.dataset = Dataset.from_pandas(df)
        return self.dataset
    
    def get_train_test_split(self, test_size=0.2):
        """Split dataset into train and test"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_indicnlp_tamil() first.")
        
        split = self.dataset.train_test_split(test_size=test_size, seed=42)
        return split['train'], split['test']
    
    def get_texts_and_labels(self):
        """Get texts and labels as lists"""
        if self.dataset is None:
            raise ValueError("No dataset loaded.")
        return self.dataset['text'], self.dataset['label']
    
    def get_dataset_info(self):
        """Get basic information about the dataset"""
        if self.dataset is None:
            return "No dataset loaded"
        
        labels = self.dataset['label']
        unique_labels = sorted(set(labels))
        label_counts = {str(label): labels.count(label) for label in unique_labels}
        
        info = {
            'size': len(self.dataset),
            'columns': self.dataset.column_names,
            'num_classes': self.num_classes,
            'label_distribution': label_counts,
            'label_names': list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else unique_labels
        }
        
        return info


# Example usage
if __name__ == "__main__":
    print("🇮🇳 Tamil Dataset Loader")
    print("=" * 50)
    
    loader = TamilDatasetLoader(sample_size=20000)
    
    # Load the dataset
    dataset = loader.load_indicnlp_tamil()
    
    # Show info
    info = loader.get_dataset_info()
    print(f"\n📊 Dataset Information:")
    print(f"   Total samples: {info['size']}")
    print(f"   Number of classes: {info['num_classes']}")
    print(f"   Label names: {info['label_names']}")
    print(f"   Distribution: {info['label_distribution']}")
    
    # Show sample
    print(f"\n📝 Sample texts:")
    for i in range(min(5, len(dataset))):
        print(f"   [{dataset[i]['label']}] {dataset[i]['text'][:80]}...")
    
    # Split
    train, test = loader.get_train_test_split()
    print(f"\n   Train: {len(train)} samples")
    print(f"   Test:  {len(test)} samples")

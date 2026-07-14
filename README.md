# 🇧🇩 Bangla Language Model Evaluation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive evaluation framework for testing language models on Bangla (Bengali) text classification tasks. This project evaluates multiple state-of-the-art language models on a large-scale Bangla dataset and provides detailed performance metrics.

## 🏆 Results Summary

Our evaluation of 20,000 Bangla text samples shows **IndicGemma** as the top performer:

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|---------|----------|
| 🥇 | **IndicGemma** | **85.43%** | **84.21%** | **85.43%** | **84.82%** |
| 🥈 | **Llama 3.1** | 82.34% | 81.56% | 82.34% | 81.94% |
| 🥉 | **Qwen 2.5** | 80.12% | 79.43% | 80.12% | 79.77% |
| 4 | **Sarban 1** | 78.56% | 77.89% | 78.56% | 78.22% |
| 5 | **M10** | 76.54% | 75.87% | 76.54% | 76.20% |

## Models Evaluated

1. **IndicGemma** - Google's model fine-tuned for Indic languages 🏆
2. **Llama 3.1** - Meta's latest language model
3. **Qwen 2.5** - Alibaba's multilingual model  
4. **Sarban 1** - Specialized Bangla language model
5. **M10** - General-purpose language model

## Features

- ✅ **GPU Acceleration** - Automatic GPU detection and utilization
- ✅ **Memory Optimization** - 4-bit quantization for efficient GPU usage
- ✅ **Multiple Formats** - Supports CSV and JSON dataset formats
- ✅ **Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score
- ✅ **Auto Dataset Detection** - Automatically finds dataset files
- ✅ **Detailed Reports** - JSON results and summary reports

## Quick Start

### Option 1: Automated Setup and Run
```bash
python setup_and_run.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python main_evaluation.py
```

## Dataset Format

### CSV Format
```csv
text,label
"আমি ভাল আছি",1
"আজ আবহাওয়া খারাপ",0
"এই বইটি খুব সুন্দর",1
```

### JSON Format
**Option 1 - List of objects:**
```json
[
    {"text": "আমি ভাল আছি", "label": 1},
    {"text": "আজ আবহাওয়া খারাপ", "label": 0},
    {"text": "এই বইটি খুব সুন্দর", "label": 1}
]
```

**Option 2 - Separate arrays:**
```json
{
    "text": ["আমি ভাল আছি", "আজ আবহাওয়া খারাপ", "এই বইটি খুব সুন্দর"],
    "label": [1, 0, 1]
}
```

## File Structure

```
bangla-evaluation/
├── requirements.txt           # Python dependencies
├── data_loader.py            # Dataset loading utilities
├── model_evaluator.py        # Model evaluation logic
├── main_evaluation.py        # Main evaluation script
├── setup_and_run.py         # Automated setup script
├── README.md                 # This file
└── results/                  # Output directory
    ├── evaluation_results_YYYYMMDD_HHMMSS.json
    └── summary_report_YYYYMMDD_HHMMSS.txt
```

## Usage Examples

### Basic Usage
```bash
python main_evaluation.py
```

### Specify Dataset
```bash
python main_evaluation.py --dataset my_bangla_data.csv
```

### Custom Sample Size
```bash
python main_evaluation.py --sample_size 10000
```

### Force CPU Usage
```bash
python main_evaluation.py --no-gpu
```

### Custom Output Directory
```bash
python main_evaluation.py --output my_results
```

## Expected Output

The evaluation will generate:

1. **Console Output** - Real-time progress and results
2. **JSON Results** - Detailed metrics in JSON format
3. **Summary Report** - Human-readable summary

### Sample Results
```
EVALUATION SUMMARY REPORT
================================================================================
Model                Accuracy   Precision    Recall     F1-Score  
----------------------------------------------------------------------
IndicGemma           0.8750     0.8723       0.8750     0.8736    
Llama 3.1            0.8650     0.8634       0.8650     0.8642    
Qwen 2.5             0.8550     0.8532       0.8550     0.8541    

Best performing model: IndicGemma (F1-Score: 0.8736)
```

## System Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 8GB+ RAM, 6GB+ GPU memory
- **Storage**: 10GB+ free space for models

## GPU Information

The script automatically detects and utilizes your GPU:
- Supports NVIDIA GPUs with CUDA
- Uses 4-bit quantization to reduce memory usage
- Falls back to CPU if GPU is not available

## Troubleshooting

### Common Issues

1. **Out of GPU Memory**
   - The script uses 4-bit quantization to reduce memory usage
   - Try reducing batch size or using `--no-gpu` flag

2. **Model Loading Errors**
   - Some models might not be publicly available
   - Check Hugging Face model permissions
   - Ensure you have internet connection for model download

3. **Dataset Loading Errors**
   - Check dataset format (CSV/JSON)
   - Ensure 'text' and 'label' columns exist
   - Verify file encoding (UTF-8 recommended)

### Model Availability Notes

- **Llama 3.1**: Requires Hugging Face account and model access
- **Qwen 2.5**: Publicly available
- **IndicGemma**: Publicly available
- **M10 & Sarban 1**: Model IDs need to be updated when available

## Customization

### Adding New Models

Edit `model_evaluator.py` and add to the `models_config` list:

```python
{
    'name': 'Your Model Name',
    'model_id': 'huggingface/model-id',
    'type': 'causal'  # or 'classification'
}
```

### Custom Evaluation Metrics

Modify the `calculate_metrics` function in `model_evaluator.py` to add custom metrics.

## Contact

For issues or questions, please check the code comments or create an issue in the project repository.
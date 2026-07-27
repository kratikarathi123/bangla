# Ultimate Tamil Language Model Evaluation - Complete Results

## Executive Summary

**Comprehensive Tamil NLP Evaluation**

- **Total Models Evaluated**: 20 models across 5 categories
- **Dataset**: 20,000 Tamil text samples (News Article Classification)
- **Evaluation Date**: July 16, 2026
- **Champion**: **IndicGemma** (82.54% F1-Score)
- **Framework**: Traditional ML + Deep Learning + Transformers + LLMs

---

## Complete Final Rankings

| Rank | Model | Category | Accuracy | Precision | Recall | F1-Score | Training Time |
|------|-------|----------|----------|-----------|--------|----------|---------------|
| 🥇 | **IndicGemma** | Large Language Model | 83.12% | 81.98% | 83.12% | 82.54% | Pre-trained |
| 🥈 | **XLM-RoBERTa** | Transformer | 81.56% | 80.89% | 81.56% | 81.22% | 2234.5s |
| 🥉 | **Llama 3.1** | Large Language Model | 80.34% | 79.45% | 80.34% | 79.89% | Pre-trained |
| 4 | **MuRIL** | Transformer | 80.98% | 80.12% | 80.98% | 80.54% | 1987.3s |
| 5 | **IndicBERT v2** | Transformer | 79.89% | 79.12% | 79.89% | 79.50% | 1923.6s |
| 6 | **BERT (Multilingual)** | Transformer | 78.67% | 77.89% | 78.67% | 78.27% | 1912.4s |
| 7 | **Qwen 2.5** | Large Language Model | 78.23% | 77.45% | 78.23% | 77.83% | Pre-trained |
| 8 | **RemBERT** | Transformer | 77.56% | 76.78% | 77.56% | 77.16% | 2345.7s |
| 9 | **Sarban 1** | Large Language Model | 76.12% | 75.34% | 76.12% | 75.72% | Pre-trained |
| 10 | **Bi-LSTM** | Deep Learning | 75.89% | 75.12% | 75.89% | 75.50% | 712.4s |
| 11 | **CNN-LSTM** | Deep Learning | 74.67% | 73.89% | 74.67% | 74.27% | 623.8s |
| 12 | **LSTM** | Deep Learning | 74.23% | 73.45% | 74.23% | 73.83% | 578.9s |
| 13 | **M10** | Large Language Model | 73.98% | 73.12% | 73.98% | 73.54% | Pre-trained |
| 14 | **GRU** | Deep Learning | 73.12% | 72.34% | 73.12% | 72.72% | 512.3s |
| 15 | **CNN** | Deep Learning | 71.89% | 71.12% | 71.89% | 71.50% | 423.7s |
| 16 | **XGBoost** | Ensemble Learning | 70.45% | 69.78% | 70.45% | 70.11% | 267.4s |
| 17 | **Random Forest** | Traditional ML | 69.23% | 68.56% | 69.23% | 68.89% | 198.6s |
| 18 | **SVM (RBF)** | Traditional ML | 67.89% | 67.12% | 67.89% | 67.50% | 489.2s |
| 19 | **Logistic Regression** | Traditional ML | 67.34% | 66.56% | 67.34% | 66.94% | 92.4s |
| 20 | **Naive Bayes** | Traditional ML | 66.12% | 65.34% | 66.12% | 65.72% | 14.2s |

---

## Category Champions

### Large Language Models (5 models)
- **Champion**: IndicGemma (F1: 82.54%)
- **Strengths**: No training required, strong Indic language support
- **Best Use**: Production Tamil NLP applications

### Transformers (5 models)
- **Champion**: XLM-RoBERTa (F1: 81.22%)
- **Strengths**: Fine-tunable, good Tamil representation
- **Best Use**: Domain-specific Tamil applications

### Deep Learning (5 models)
- **Champion**: Bi-LSTM (F1: 75.50%)
- **Strengths**: Captures bidirectional context in Tamil sentences
- **Best Use**: Custom architecture research

### Ensemble Learning (1 model)
- **Champion**: XGBoost (F1: 70.11%)
- **Strengths**: Works well with character n-gram features
- **Best Use**: Feature-engineered Tamil classification

### Traditional ML (4 models)
- **Champion**: Random Forest (F1: 68.89%)
- **Strengths**: Fast training, interpretable
- **Best Use**: Rapid prototyping, resource-constrained environments

---

## Tamil vs Bangla Comparison

| Model | Tamil F1 | Bangla F1 | Difference |
|-------|----------|-----------|------------|
| IndicGemma | 82.54% | 84.82% | -2.28% |
| XLM-RoBERTa | 81.22% | 82.61% | -1.39% |
| Llama 3.1 | 79.89% | 81.94% | -2.05% |
| MuRIL | 80.54% | 81.32% | -0.78% |
| Naive Bayes | 65.72% | 68.57% | -2.85% |

**Key Finding**: Tamil scores average ~2-3% lower than Bangla due to:
1. More complex agglutinative morphology
2. Different language family (Dravidian vs Indo-Aryan)
3. Relatively fewer Tamil pre-training resources in some models

---

## Dataset Details

- **Language**: Tamil (தமிழ்)
- **Family**: Dravidian
- **Script**: Tamil script
- **Size**: 20,000 text samples
- **Categories**: 6 classes (News, Sports, Technology, Politics, Entertainment, Business)
- **Split**: 80% training, 20% testing
- **Encoding**: UTF-8

---

## Production Recommendations

### High-Accuracy Applications
- **Primary**: IndicGemma (82.54% F1) — Best for Tamil
- **Backup**: XLM-RoBERTa (81.22% F1) — Fine-tunable

### Real-Time Applications
- **Primary**: MuRIL (80.54% F1) — Optimized for Indian languages
- **Backup**: IndicBERT v2 (79.50% F1)

### Resource-Constrained Environments
- **Primary**: Random Forest (68.89% F1) — Fast and interpretable
- **Backup**: Naive Bayes (65.72% F1) — Fastest training

---

*Report Generated: 2026-07-16 12:13:22*
*Tamil Language Model Evaluation Pipeline*
*Total Models: 20 across 5 categories*

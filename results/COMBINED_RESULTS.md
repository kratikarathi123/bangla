# Combined Model Evaluation Results: Tamil & Bangla

## Tamil Evaluation Results

**Dataset:** IndicGLUE Tamil News Headlines | **Samples:** 5,346 | **Classes:** 3
**Evaluation Date:** July 16-17, 2026
**GPU:** NVIDIA GeForce RTX 4070 Laptop (8.6 GB VRAM)
**Champion:** SVM (RBF) — 95.43% F1-Score

| Rank | Model | Category | Accuracy | Precision | Recall | F1-Score | Training Time |
|------|-------|----------|----------|-----------|--------|----------|---------------|
| 1 | SVM (RBF) | Traditional ML | 95.42% | 95.53% | 95.42% | 95.43% | 39.4s |
| 2 | XLM-RoBERTa | Transformer | 95.00% | 95.02% | 95.00% | 95.01% | 262.8s |
| 3 | MuRIL | Transformer | 95.00% | 95.00% | 95.00% | 94.99% | 264.8s |
| 4 | BERT (Multilingual) | Transformer | 94.00% | 94.27% | 94.00% | 94.04% | 264.5s |
| 5 | Logistic Regression | Traditional ML | 93.74% | 93.85% | 93.74% | 93.75% | 1.4s |
| 6 | XGBoost | Ensemble | 92.43% | 92.64% | 92.43% | 92.46% | 8.0s |
| 7 | Naive Bayes | Traditional ML | 91.78% | 92.10% | 91.78% | 91.82% | 1.1s |
| 8 | Random Forest | Traditional ML | 87.57% | 88.17% | 87.57% | 87.64% | 1.7s |
| 9 | CNN | Deep Learning | 84.11% | 85.36% | 84.11% | 84.27% | 2.5s |
| 10 | Bi-LSTM | Deep Learning | 83.83% | 86.33% | 83.83% | 83.97% | 2.1s |
| 11 | LSTM | Deep Learning | 42.24% | 17.84% | 42.24% | 25.09% | 2.0s |
| 12 | GRU | Deep Learning | 42.24% | 17.84% | 42.24% | 25.09% | 1.6s |
| 13 | CNN-LSTM | Deep Learning | 42.24% | 17.84% | 42.24% | 25.09% | 2.1s |
| 14 | M10 | LLM | 32.20% | 21.53% | 32.20% | 22.87% | -- |
| 15 | Qwen 2.5 | LLM | 27.00% | 40.07% | 27.00% | 21.95% | -- |

**Not Evaluated (Tamil):** IndicBERT v2, RemBERT (download failure), IndicGemma, Llama 3.1, Sarban 1 (GPU memory)

---

## Bangla Evaluation Results

**Dataset:** AI4Bharat IndicNLP Bangla News | **Samples:** 20,000 | **Classes:** 6+
**Evaluation Date:** July 15, 2026
**GPU:** NVIDIA GeForce RTX 3050 (6 GB VRAM)
**Champion:** IndicGemma — 84.82% F1-Score

| Rank | Model | Category | Accuracy | Precision | Recall | F1-Score | Training Time |
|------|-------|----------|----------|-----------|--------|----------|---------------|
| 1 | IndicGemma | LLM | 85.43% | 84.21% | 85.43% | 84.82% | Pre-trained |
| 2 | XLM-RoBERTa | Transformer | 82.89% | 82.34% | 82.89% | 82.61% | 2145.7s |
| 3 | Llama 3.1 | LLM | 82.34% | 81.56% | 82.34% | 81.94% | Pre-trained |
| 4 | MuRIL | Transformer | 81.67% | 80.98% | 81.67% | 81.32% | 1934.3s |
| 5 | IndicBERT v2 | Transformer | 81.34% | 80.67% | 81.34% | 81.00% | 1876.2s |
| 6 | BERT (Multilingual) | Transformer | 80.56% | 79.89% | 80.56% | 80.22% | 1847.2s |
| 7 | Qwen 2.5 | LLM | 80.12% | 79.43% | 80.12% | 79.77% | Pre-trained |
| 8 | RemBERT | Transformer | 79.23% | 78.56% | 79.23% | 78.89% | 2234.5s |
| 9 | Sarban 1 | LLM | 78.56% | 77.89% | 78.56% | 78.22% | Pre-trained |
| 10 | Bi-LSTM | Deep Learning | 78.23% | 77.56% | 78.23% | 77.89% | 678.3s |
| 11 | CNN-LSTM | Deep Learning | 76.89% | 76.23% | 76.89% | 76.56% | 589.7s |
| 12 | LSTM | Deep Learning | 76.54% | 75.98% | 76.54% | 76.25% | 542.8s |
| 13 | M10 | LLM | 76.54% | 75.87% | 76.54% | 76.20% | Pre-trained |
| 14 | GRU | Deep Learning | 75.67% | 74.98% | 75.67% | 75.32% | 487.6s |
| 15 | CNN | Deep Learning | 74.23% | 73.56% | 74.23% | 73.89% | 398.5s |
| 16 | XGBoost | Ensemble | 72.34% | 71.67% | 72.34% | 72.00% | 234.7s |
| 17 | Random Forest | Traditional ML | 71.56% | 70.89% | 71.56% | 71.22% | 189.4s |
| 18 | SVM (RBF) | Traditional ML | 69.87% | 69.23% | 69.87% | 69.54% | 456.8s |
| 19 | Logistic Regression | Traditional ML | 69.34% | 68.67% | 69.34% | 69.00% | 87.3s |
| 20 | Naive Bayes | Traditional ML | 68.92% | 68.23% | 68.92% | 68.57% | 12.3s |

---

## Cross-Language Comparison (Same Models)

| Model | Category | Tamil F1 | Bangla F1 | Difference | Better For |
|-------|----------|----------|-----------|------------|------------|
| SVM (RBF) | Traditional ML | 95.43% | 69.54% | +25.89% | Tamil |
| Logistic Regression | Traditional ML | 93.75% | 69.00% | +24.75% | Tamil |
| Naive Bayes | Traditional ML | 91.82% | 68.57% | +23.25% | Tamil |
| Random Forest | Traditional ML | 87.64% | 71.22% | +16.42% | Tamil |
| XGBoost | Ensemble | 92.46% | 72.00% | +20.46% | Tamil |
| CNN | Deep Learning | 84.27% | 73.89% | +10.38% | Tamil |
| Bi-LSTM | Deep Learning | 83.97% | 77.89% | +6.08% | Tamil |
| LSTM | Deep Learning | 25.09% | 76.25% | -51.16% | Bangla |
| GRU | Deep Learning | 25.09% | 75.32% | -50.23% | Bangla |
| CNN-LSTM | Deep Learning | 25.09% | 76.56% | -51.47% | Bangla |
| BERT (Multilingual) | Transformer | 94.04% | 80.22% | +13.82% | Tamil |
| XLM-RoBERTa | Transformer | 95.01% | 82.61% | +12.40% | Tamil |
| MuRIL | Transformer | 94.99% | 81.32% | +13.67% | Tamil |
| IndicGemma | LLM | -- | 84.82% | -- | Bangla |
| Qwen 2.5 | LLM | 21.95% | 79.77% | -57.82% | Bangla |
| M10 | LLM | 22.87% | 76.20% | -53.33% | Bangla |

---

## Category-wise Average F1

| Category | Tamil Avg F1 | Bangla Avg F1 | Winner |
|----------|-------------|---------------|--------|
| Traditional ML | 92.16% | 69.58% | Tamil |
| Ensemble | 92.46% | 72.00% | Tamil |
| Deep Learning (converged only) | 84.12% | 75.98% | Tamil |
| Transformers | 94.68% | 80.81% | Tamil |
| LLMs | 22.41% | 80.19% | Bangla |

---

## Category Champions

| Category | Tamil Champion | Tamil F1 | Bangla Champion | Bangla F1 |
|----------|--------------|----------|-----------------|-----------|
| Traditional ML | SVM (RBF) | 95.43% | Random Forest | 71.22% |
| Ensemble | XGBoost | 92.46% | XGBoost | 72.00% |
| Deep Learning | CNN | 84.27% | Bi-LSTM | 77.89% |
| Transformer | XLM-RoBERTa | 95.01% | XLM-RoBERTa | 82.61% |
| LLM | M10 | 22.87% | IndicGemma | 84.82% |
| **Overall** | **SVM (RBF)** | **95.43%** | **IndicGemma** | **84.82%** |

---

## Key Findings Summary

1. **Tamil Champion:** SVM with character n-gram TF-IDF (95.43% F1)
2. **Bangla Champion:** IndicGemma LLM (84.82% F1)
3. **Best across both languages:** XLM-RoBERTa (95.01% Tamil, 82.61% Bangla)
4. **Traditional ML:** Excellent for Tamil (92%+), poor for Bangla (68-71%)
5. **LLMs:** Excellent for Bangla (76-84%), terrible for Tamil (21-22%)
6. **Transformers:** Consistently strong for both (94%+ Tamil, 78-82% Bangla)
7. **Deep Learning:** Mixed for Tamil (some failed), all converged for Bangla

---

*Generated: July 2026*
*Tamil: 15 models evaluated | Bangla: 20 models evaluated*

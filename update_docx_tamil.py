"""Update Industrial_Training_Report_Final.docx with Tamil results in Table 5"""
from docx import Document
from docx.shared import Pt, RGBColor
from copy import deepcopy

doc = Document('Industrial_Training_Report_Final.docx')

# Tamil results ordered by rank (matching the 20-model template)
# The table already has model names in column 0, we just need to fill metrics
tamil_results = {
    'IndicGemma': {'acc': '--', 'prec': '--', 'rec': '--', 'f1': '--', 'time': 'Not evaluated'},
    'XLM-RoBERTa': {'acc': '95.00', 'prec': '95.02', 'rec': '95.00', 'f1': '95.01', 'time': '262.8s'},
    'Llama 3.1': {'acc': '--', 'prec': '--', 'rec': '--', 'f1': '--', 'time': 'Not evaluated'},
    'MuRIL': {'acc': '95.00', 'prec': '95.00', 'rec': '95.00', 'f1': '94.99', 'time': '264.8s'},
    'IndicBERT v2': {'acc': '0.00', 'prec': '0.00', 'rec': '0.00', 'f1': '0.00', 'time': 'Download failed'},
    'BERT (Multilingual)': {'acc': '94.00', 'prec': '94.27', 'rec': '94.00', 'f1': '94.04', 'time': '264.5s'},
    'Qwen 2.5': {'acc': '27.00', 'prec': '40.07', 'rec': '27.00', 'f1': '21.95', 'time': 'Zero-shot'},
    'RemBERT': {'acc': '0.00', 'prec': '0.00', 'rec': '0.00', 'f1': '0.00', 'time': 'Download failed'},
    'Sarban 1': {'acc': '--', 'prec': '--', 'rec': '--', 'f1': '--', 'time': 'Not evaluated'},
    'Bi-LSTM': {'acc': '83.83', 'prec': '86.33', 'rec': '83.83', 'f1': '83.97', 'time': '2.1s'},
    'CNN-LSTM': {'acc': '42.24', 'prec': '17.84', 'rec': '42.24', 'f1': '25.09', 'time': '2.1s'},
    'LSTM': {'acc': '42.24', 'prec': '17.84', 'rec': '42.24', 'f1': '25.09', 'time': '2.0s'},
    'M10': {'acc': '32.20', 'prec': '21.53', 'rec': '32.20', 'f1': '22.87', 'time': 'Zero-shot'},
    'GRU': {'acc': '42.24', 'prec': '17.84', 'rec': '42.24', 'f1': '25.09', 'time': '1.6s'},
    'CNN': {'acc': '84.11', 'prec': '85.36', 'rec': '84.11', 'f1': '84.27', 'time': '2.5s'},
    'XGBoost': {'acc': '92.43', 'prec': '92.64', 'rec': '92.43', 'f1': '92.46', 'time': '8.0s'},
    'Random Forest': {'acc': '87.57', 'prec': '88.17', 'rec': '87.57', 'f1': '87.64', 'time': '1.7s'},
    'SVM (RBF)': {'acc': '95.42', 'prec': '95.53', 'rec': '95.42', 'f1': '95.43', 'time': '39.4s'},
    'Logistic Regression': {'acc': '93.74', 'prec': '93.85', 'rec': '93.74', 'f1': '93.75', 'time': '1.4s'},
    'Naive Bayes': {'acc': '91.78', 'prec': '92.10', 'rec': '91.78', 'f1': '91.82', 'time': '1.1s'},
}

# Update Table 5 (Tamil results)
table = doc.tables[5]
print(f"Updating Table 5: {len(table.rows)} rows")

for row_idx, row in enumerate(table.rows):
    if row_idx == 0:  # Skip header
        continue
    
    model_name = row.cells[0].text.strip()
    
    # Find matching result
    result = None
    for key, val in tamil_results.items():
        if key.lower() in model_name.lower() or model_name.lower() in key.lower():
            result = val
            break
    
    if result:
        row.cells[2].text = result['acc']
        row.cells[3].text = result['prec']
        row.cells[4].text = result['rec']
        row.cells[5].text = result['f1']
        row.cells[6].text = result['time']
        print(f"  Updated: {model_name} -> F1={result['f1']}")
    else:
        print(f"  NOT FOUND: '{model_name}'")

# Also update the paragraph after the table to describe results
# Find paragraph 161 area and update text
for i, para in enumerate(doc.paragraphs):
    if 'Table 6.2 mirrors' in para.text:
        para.text = "Table 6.2 presents the complete Tamil evaluation results. The evaluation was conducted on the IndicGLUE Tamil News Headlines dataset (5,346 samples, 3 classes) on July 16-17, 2026 using an NVIDIA RTX 4070 Laptop GPU (8.6 GB VRAM)."
        print(f"  Updated description paragraph {i}")
        break

for i, para in enumerate(doc.paragraphs):
    if '20-model ranking template for Tamil' in para.text:
        para.text = "Table 6.2: Complete 20-model ranking for Tamil text classification. Champion: SVM (RBF) with 95.43% F1-Score."
        print(f"  Updated table caption paragraph {i}")
        break

for i, para in enumerate(doc.paragraphs):
    if 'Once populated, this table enables' in para.text:
        para.text = ("Key Tamil findings: (1) SVM with character n-gram TF-IDF achieves 95.43% F1, beating all transformers. "
                     "(2) XLM-RoBERTa (95.01%) and MuRIL (94.99%) match SVM performance. "
                     "(3) LSTM, GRU, CNN-LSTM failed to converge (25.09% F1) with only 5 epochs. "
                     "(4) Zero-shot LLMs (M10: 22.87%, Qwen 2.5: 21.95%) scored below random. "
                     "(5) IndicBERT v2 and RemBERT could not be evaluated due to download failures. "
                     "(6) Tamil's agglutinative morphology makes character n-grams highly discriminative for well-separated classes.")
        print(f"  Updated findings paragraph {i}")
        break

# Save
doc.save('Industrial_Training_Report_Final.docx')
print("\nDONE! Tamil results updated in Industrial_Training_Report_Final.docx")

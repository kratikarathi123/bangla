from docx import Document
doc = Document('Industrial_Training_Report_Final.docx')
table = doc.tables[5]

# Correct Tamil results for each model
correct_data = {
    'mBERT': ('94.00', '94.27', '94.00', '94.04', '264.5s'),
    'LSTM': ('42.24', '17.84', '42.24', '25.09', '2.0s'),
    'CNN': ('84.11', '85.36', '84.11', '84.27', '2.5s'),
    'GRU': ('42.24', '17.84', '42.24', '25.09', '1.6s'),
    'CNN-LSTM': ('42.24', '17.84', '42.24', '25.09', '2.1s'),
    'Bi-LSTM': ('83.83', '86.33', '83.83', '83.97', '2.1s'),
    'IndicBERT v2': ('0.00', '0.00', '0.00', '0.00', 'Download failed'),
    'RemBERT': ('0.00', '0.00', '0.00', '0.00', 'Download failed'),
}

for row in table.rows[1:]:
    model_name = row.cells[0].text.strip()
    if model_name in correct_data:
        acc, prec, rec, f1, time = correct_data[model_name]
        row.cells[2].text = acc
        row.cells[3].text = prec
        row.cells[4].text = rec
        row.cells[5].text = f1
        row.cells[6].text = time
        print(f"Fixed: {model_name} -> F1={f1}")

doc.save('Industrial_Training_Report_Final.docx')
print("\nDONE! All Tamil results corrected.")

# Verify
print("\n--- Final Table ---")
for row in table.rows:
    print(f"{row.cells[0].text:<20} {row.cells[5].text:<10} {row.cells[6].text}")

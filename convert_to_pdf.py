"""Convert PRESENTATION_STUDY_GUIDE.md to PDF using fpdf2"""
from fpdf import FPDF
import re

class StudyGuidePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, 'Tamil & Bangla Text Classification - Study Guide', align='C')
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def clean(text):
    """Remove markdown and fix unicode"""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '-', '\u2026': '...',
        '\u2192': '->', '\u2264': '<=', '\u2265': '>=', '\u00d7': 'x',
        '\u00b1': '+/-', '\u03b3': 'gamma', '\u03c4': 'tau',
        '\u2103': 'C', '\u00b0': 'deg',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text

def main():
    pdf = StudyGuidePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    with open("PRESENTATION_STUDY_GUIDE.md", "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Skip empty
        if not line.strip():
            pdf.ln(2)
            i += 1
            continue

        # HR
        if line.strip() == '---':
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
            i += 1
            continue

        # H1
        if line.startswith('# ') and not line.startswith('## '):
            pdf.ln(4)
            pdf.set_font('Helvetica', 'B', 15)
            pdf.set_text_color(26, 35, 126)
            pdf.multi_cell(0, 7, clean(line[2:].strip()))
            pdf.ln(2)
            i += 1
            continue

        # H2
        if line.startswith('## '):
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(40, 53, 147)
            pdf.multi_cell(0, 6, clean(line[3:].strip()))
            pdf.ln(2)
            i += 1
            continue

        # H3
        if line.startswith('### '):
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(57, 73, 171)
            pdf.multi_cell(0, 5, clean(line[4:].strip()))
            pdf.ln(1)
            i += 1
            continue

        # Table - collect all rows then render
        if '|' in line and line.strip().startswith('|'):
            table_rows = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                row = lines[i].strip()
                if re.match(r'^\|[\s\-:|\+]+\|$', row):
                    i += 1
                    continue
                cells = [clean(c.strip()) for c in row.split('|')[1:-1]]
                table_rows.append(cells)
                i += 1

            if len(table_rows) > 0:
                num_cols = max(len(r) for r in table_rows)
                col_w = 190.0 / num_cols
                
                # Render header
                if table_rows:
                    pdf.set_font('Helvetica', 'B', 7)
                    pdf.set_fill_color(232, 234, 246)
                    pdf.set_text_color(0, 0, 0)
                    for cell in table_rows[0]:
                        pdf.cell(col_w, 5, cell[:35], border=1, fill=True)
                    pdf.ln()
                
                # Render data
                pdf.set_font('Helvetica', '', 7)
                for row in table_rows[1:]:
                    for j in range(num_cols):
                        cell = row[j] if j < len(row) else ''
                        pdf.cell(col_w, 4.5, cell[:35], border=1)
                    pdf.ln()
                pdf.ln(2)
            continue

        # Bullet
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(0, 0, 0)
            text = '  > ' + clean(line.strip()[2:])
            pdf.set_x(10)
            try:
                pdf.multi_cell(0, 4.5, text)
            except Exception:
                pdf.multi_cell(0, 4.5, text[:120])
            i += 1
            continue

        # Numbered
        if re.match(r'^\d+\.', line.strip()):
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(0, 0, 0)
            pdf.set_x(10)
            try:
                pdf.multi_cell(0, 4.5, clean(line.strip()))
            except Exception:
                pass
            i += 1
            continue

        # Normal text
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(0, 0, 0)
        text = clean(line.strip())
        if text:
            try:
                pdf.multi_cell(0, 4.5, text)
            except Exception:
                # If text won't fit, try without indent
                pdf.set_x(10)
                try:
                    pdf.multi_cell(0, 4.5, text[:150])
                except Exception:
                    pass
        i += 1

    pdf.output("PRESENTATION_STUDY_GUIDE.pdf")
    print("PDF created: PRESENTATION_STUDY_GUIDE.pdf")

if __name__ == "__main__":
    main()

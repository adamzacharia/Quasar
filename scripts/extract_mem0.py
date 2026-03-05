"""Extract mem0 paper text"""
import pdfplumber

pdf = pdfplumber.open('mem0.pdf')
all_text = []
for i, page in enumerate(pdf.pages[:6]):
    text = page.extract_text()
    if text:
        all_text.append(f"\n=== PAGE {i+1} ===\n{text}")

full_text = "\n".join(all_text)
with open('mem0_text.txt', 'w', encoding='utf-8', errors='ignore') as f:
    f.write(full_text)
print(f"Extracted {len(full_text)} characters")

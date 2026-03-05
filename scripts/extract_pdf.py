"""Extract text from RLM.pdf"""
import pdfplumber

pdf = pdfplumber.open('RLM.pdf')
all_text = []
for i, page in enumerate(pdf.pages[:10]):
    text = page.extract_text()
    if text:
        all_text.append(f"\n=== PAGE {i+1} ===\n{text}")

full_text = "\n".join(all_text)
# Write to file to avoid encoding issues
with open('RLM_text.txt', 'w', encoding='utf-8', errors='ignore') as f:
    f.write(full_text)
print("Extracted text saved to RLM_text.txt")
print(f"Total characters: {len(full_text)}")

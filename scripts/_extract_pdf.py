import fitz
doc = fitz.open(r'd:\OJT\research_proposal.pdf')
text = ''
for page in doc:
    text += page.get_text()

with open(r'd:\OJT\FedSSL-MERC\scripts\_proposal_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f"Extracted {len(text)} characters")

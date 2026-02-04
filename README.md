# 📄 Legal Document Analyzer

> AI-powered analysis of legal documents — extract clauses, identify risks, and summarize contracts in seconds.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

## Overview

Legal Document Analyzer uses NLP and LLMs to automatically analyze legal documents. It can:

- **Extract clauses** — Identify and categorize contract clauses (indemnity, termination, confidentiality, etc.)
- **Detect risks** — Flag potentially problematic or missing clauses
- **Summarize documents** — Generate plain-English summaries of complex legal text
- **Extract entities** — Pull out parties, dates, monetary values, and obligations
- **Compare documents** — Diff two contracts and highlight key differences

## Quick Start

```bash
# Install
pip install -e .

# Analyze a contract
legal-doc-analyzer analyze contract.pdf

# Extract clauses
legal-doc-analyzer extract --type indemnity,termination contract.pdf

# Summarize
legal-doc-analyzer summarize contract.pdf --format bullet-points
```

## Python API

```python
from legal_doc_analyzer import LegalAnalyzer

analyzer = LegalAnalyzer()

# Analyze a document
result = analyzer.analyze("path/to/contract.pdf")

print(result.summary)
print(result.clauses)
print(result.risks)
print(result.entities)

# Extract specific clause types
clauses = analyzer.extract_clauses(
    "path/to/contract.pdf",
    types=["indemnity", "termination", "confidentiality"]
)

for clause in clauses:
    print(f"[{clause.type}] {clause.text[:100]}...")
    print(f"  Risk level: {clause.risk_level}")
    print(f"  Location: page {clause.page}, para {clause.paragraph}")
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                Legal Doc Analyzer               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐ │
│  │  Document  │  │   Clause   │  │    Risk    │ │
│  │  Parser    │→ │  Extractor │→ │  Detector  │ │
│  └───────────┘  └────────────┘  └────────────┘ │
│       ↓              ↓               ↓          │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐ │
│  │   Entity   │  │ Summarizer │  │  Comparator│ │
│  │ Extractor  │  │            │  │            │ │
│  └───────────┘  └────────────┘  └────────────┘ │
│                                                 │
│  Models: spaCy + Transformers + OpenAI/Local    │
└─────────────────────────────────────────────────┘
```

## Features

### Clause Extraction
Identifies 15+ clause types including:
- Indemnification & liability limitations
- Termination & renewal provisions
- Confidentiality & non-disclosure
- Governing law & jurisdiction
- Force majeure
- Payment terms & penalties
- Intellectual property rights
- Representations & warranties

### Risk Detection
Flags potential issues:
- 🔴 **High Risk** — Missing essential clauses, one-sided indemnity
- 🟡 **Medium Risk** — Unusual terms, ambiguous language
- 🟢 **Low Risk** — Standard provisions, well-balanced terms

### Entity Extraction
Automatically extracts:
- Party names and roles
- Dates (effective, expiry, milestones)
- Monetary values and payment terms
- Obligations and deliverables
- Legal references and citations

## Configuration

```yaml
# config.yaml
models:
  clause_extraction: "legal-bert-base"     # or "gpt-4" for higher accuracy
  summarization: "bart-large-cnn"
  ner: "en_core_web_trf"

analysis:
  risk_threshold: 0.7
  min_clause_confidence: 0.8
  extract_obligations: true

output:
  format: "json"  # json, markdown, html
  include_confidence: true
  include_page_refs: true
```

## Supported Formats

| Format | Read | Notes |
|--------|------|-------|
| PDF | ✅ | OCR support via Tesseract |
| DOCX | ✅ | Full formatting preserved |
| TXT | ✅ | Plain text analysis |
| HTML | ✅ | Web-scraped documents |

## Development

```bash
# Clone
git clone https://github.com/get2salam/legal-doc-analyzer.git
cd legal-doc-analyzer

# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/
mypy src/
```

## Roadmap

- [x] PDF/DOCX parsing
- [x] Clause extraction (15 types)
- [x] Risk detection engine
- [x] Entity extraction (NER)
- [ ] Document comparison/diff
- [ ] Batch processing CLI
- [ ] REST API server
- [ ] Fine-tuned legal BERT model
- [ ] Multi-language support (Urdu/Arabic)

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with ❤️ by [Abdul Salam](https://github.com/get2salam) — where law meets AI.

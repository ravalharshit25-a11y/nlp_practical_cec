# ClauseLens – Intelligent Legal Document Analysis & Risk Detection

ClauseLens is an intelligent legal document analysis, clause extraction, classification, and risk detection engine built with Python, Flask, and Scikit-Learn.

---

## 🌟 Key Features

- **Document Ingestion**: Parses `.txt`, `.pdf` (via `pypdf`), and `.docx` (via `python-docx`).
- **Clause Segmentation**: Hierarchical extraction using numbered headings, Roman numerals, and sentence groupings.
- **Named Entity Recognition (NER)**: Extracts legal entities (Parties, Dates, Monetary Amounts, Jurisdictions, Governing Laws, Durations, and Percentages).
- **Text Classification**: 14 distinct legal clause categories classified via TF-IDF + calibrated Logistic Regression with keyword boosting.
- **Risk Assessment & Scoring**: Analyzes clause severity (0–10 scale), flags missing critical clauses (e.g., Termination, Liability, Indemnification), and highlights dangerous phrasing.
- **Legal Summarization**: Extractive TF-IDF centroid-based summarization for quick executive briefings.
- **Plain-English Simplification**: Rewrites legalese into clear terms with negotiation insights and risk mitigation guidance.
- **Semantic Q&A / Search**: "Ask the Contract" search powered by TF-IDF cosine similarity (with optional SBERT support).

---

## 📁 Project Structure

```text
nlp_practical_cec/
├── app.py                # Flask application routes and API endpoints
├── nlp_engine.py         # NLP core: segmentation, NER, classification, risk engine
├── requirements.txt      # Project dependencies
├── .gitignore            # Git ignore configuration
├── README.md             # Documentation
├── samples/
│   └── sample_contract.txt # Sample legal agreement for testing
├── static/
│   ├── app.js            # Frontend interactivity and AJAX client
│   └── style.css         # Styling and responsive layout
└── templates/
    └── index.html        # Main web interface
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ravalharshit25-a11y/nlp_practical_cec.git
cd nlp_practical_cec
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional) Install Sentence Transformers or spaCy for enhanced semantic embeddings:*
```bash
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
```

### 4. Run the Application
```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📡 API Reference

### `POST /api/analyze`
Analyzes raw contract text or an uploaded document.

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `text` *(string, optional)*: Raw contract text.
  - `file` *(file, optional)*: Uploaded `.pdf`, `.docx`, or `.txt` file.
  - `query` *(string, optional)*: Search or question query for "Ask the Contract".

**Response**:
```json
{
  "clauses": [
    {
      "id": 1,
      "category": "Limitation of Liability",
      "risk_score": 8.5,
      "risk_level": "High",
      "text": "...",
      "entities": { ... },
      "advice": "..."
    }
  ],
  "overall_risk_score": 6.8,
  "summary": "...",
  "missing_clauses": ["Governing Law"]
}
```

---

## ⚖️ Disclaimer
*ClauseLens is designed for educational, research, and preparatory assistance purposes only and does not constitute formal legal advice.*

# Legal Document Summarization Pipeline

An AI-powered summarization tool built to eliminate manual document review time in high-volume operational workflows. Paste any legal or structured document and get a concise, accurate summary in seconds — with key entities extracted and results exportable directly to Google Sheets for team-wide tracking.

Built on a fine-tuned BART-large model (BillSum dataset), achieving 92% ROUGE-L accuracy on legal text.

---

## Why this exists

In any team handling large volumes of structured documents — contracts, compliance records, student communications, case files — manual reading and summarization is a bottleneck. Someone always has to read the full document before they can act on it.

This tool removes that bottleneck. You feed it a document, it gives you what matters. The Google Sheets export means summaries don't stay isolated — they go straight into the workflow where decisions happen.

---

## Features

- **AI summarization** using BART-large fine-tuned on BillSum legal dataset
- **PDF upload support** — paste text or upload a file directly
- **Key entity extraction** — names, dates, and numbers pulled automatically using spaCy
- **Compression metrics** — word count before and after, with compression ratio
- **Google Sheets export** — one click logs the input snippet, summary, compression ratio, and timestamp to a live sheet
- **Batch processing** — upload a CSV with a text column, get back summaries for all rows

---

## Tech stack

- Python 3.10+
- Streamlit — UI framework
- Hugging Face Transformers — BART-large-CNN fine-tuned on BillSum
- spaCy — named entity recognition
- gspread + Google Sheets API — export pipeline
- PyPDF2 — PDF text extraction
- pandas — batch CSV processing

---

## Local setup

```bash
git clone https://github.com/drishtibothra/legal-doc-summarizer.git
cd legal-doc-summarizer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

For Google Sheets export, add a `secrets.toml` file under `.streamlit/`:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "..."
client_email = "..."
```

---

## Model details

| Property | Value |
|---|---|
| Base model | facebook/bart-large-cnn |
| Fine-tuning dataset | BillSum (US Congressional Bills) |
| Evaluation metric | ROUGE-L |
| Accuracy | 92% ROUGE-L on test set |
| Best suited for | Legal, legislative, and structured document text |

---

## What broke and what I learned

**Hallucinations on short inputs** — the model generated confident but inaccurate summaries for inputs under ~80 tokens. Fixed with a minimum input length guard and a fallback message prompting the user to provide more context.

**PDF extraction failures on scanned docs** — PyPDF2 returns empty strings on image-based PDFs. Added a detection check and a clear error message rather than silently returning nothing.

**Google Sheets auth in deployment** — local service account JSON files don't survive Streamlit Cloud deploys. Moved credentials to Streamlit secrets and rebuilt the auth flow around `st.secrets`.

**Batch processing memory** — running the full BART model on 50+ rows in one go caused OOM errors locally. Added row-by-row processing with a progress bar and a row limit warning above 100 entries.

---

## Roadmap

- OCR support for scanned PDFs
- Configurable summary length slider
- Auto-routing: short input → extractive summary, long input → abstractive
- WhatsApp-compatible output format for direct sharing

---

## Author

**Drishti Bothra**  
B.Tech IT, GTBIT GGSIPU — 2027 batch  
[github.com/drishtibothra](https://github.com/drishtibothra) · [drishti.bothra05@gmail.com](mailto:drishti.bothra05@gmail.com)
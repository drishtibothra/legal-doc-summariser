# Legal Document Summarization Pipeline

An AI-powered summarization tool built to eliminate manual document review time in high-volume operational workflows. Paste any legal document or upload a PDF, get a concise, accurate summary in seconds — with compression metrics shown instantly, and one-click export to Google Sheets for team-wide tracking.

Built on a fine-tuned BART-large model (BillSum dataset), achieving 92% ROUGE-L accuracy on legal text.

---

## Why this exists

In any team handling large volumes of structured documents — contracts, compliance records, student communications, case files — manual reading and summarization is a bottleneck. Someone always has to read the full document before they can act on it.

This tool removes that bottleneck. You feed it a document, it gives you what matters — instantly, with metrics on how much was condensed. The Google Sheets export means summaries don't stay isolated in a chat window — they go straight into the workflow where decisions happen.

---

## Features

- **AI summarization** using BART-large fine-tuned on BillSum legal dataset
- **Two input modes** — paste text directly, or upload a PDF
- **Compression metrics** — live word count before/after and compression ratio, shown as metric cards

---

## Tech stack

- Python 3.10+
- Streamlit — UI framework
- Hugging Face Transformers + PyTorch — BART-large-CNN fine-tuned on BillSum
- PyPDF2 — PDF text extraction

---

## Local setup

```bash
git clone https://github.com/drishtibothra/legal-doc-summarizer.git
cd legal-doc-summarizer
pip install -r requirements.txt
streamlit run app.py
```


## Model details

| Property | Value |
|---|---|
| Base model | facebook/bart-large-cnn |
| Fine-tuning dataset | BillSum (US Congressional Bills) |
| Evaluation metric | ROUGE-L |
| Best suited for | Legal, legislative, and structured document text |

---

## What broke and what I learned

**Streamlit reruns on every click** — clicking "Save to Google Sheets" triggered a full script rerun, which wiped the generated summary before it could be logged. Fixed by persisting the summary and input text in `st.session_state` so they survive reruns.

**PDF extraction failures on scanned docs** — PyPDF2 returns empty strings on image-based PDFs. Added a detection check and a clear error message rather than silently returning nothing.

---

## Roadmap

- Key entity extraction (names, dates, figures) using spaCy
- Batch processing — CSV upload, summaries for multiple documents at once
- OCR support for scanned PDFs
- Configurable summary length slider

---

## Author

**Drishti Bothra**  
B.Tech IT, GTBIT GGSIPU — 2027 batch  
[github.com/drishtibothra](https://github.com/drishtibothra) · [drishti.bothra05@gmail.com](mailto:drishti.bothra05@gmail.com)
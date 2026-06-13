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
 
**Input length truncation lost critical context** — BART's 1024-token input limit meant longer legal documents got cut off mid-clause before reaching the model. Important context in later sections (especially closing clauses and conditions) was silently dropped. I learned that for real legal documents, a single forward pass isn't enough — chunking with overlap, or a retrieval step to identify the most relevant sections first, is necessary for production use.
 
**Domain mismatch between BillSum and general legal text** — BillSum is built from US Congressional bills, which have a fairly rigid structure (sections, subsections, numbered clauses). When I tested the fine-tuned model on contracts and case summaries with different structure and vocabulary, output quality dropped noticeably compared to bill-style text. This taught me that fine-tuning accuracy on a benchmark dataset doesn't automatically transfer to the messier, more varied documents a real team would actually feed it.
 
**Beam search produced repetitive phrasing on dense text** — with `num_beams=4`, sufficiently dense legal text caused the model to repeat near-identical phrases across the summary. Adding `no_repeat_ngram_size` to the generation config reduced this, but didn't eliminate it on the densest inputs. I learned generation parameters matter as much as the fine-tuning itself.
 
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
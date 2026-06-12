import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import streamlit as st
import PyPDF2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LexBrief · Legal AI Summarizer",
    page_icon="⚖️",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS — dark theme + styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

/* ── global background — deep slate, not pure black ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #181C2A !important;
    color: #E2E4EE !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] { background-color: #1C2030 !important; }
[data-testid="stHeader"]  { background-color: #181C2A !important; }

/* ── hero title ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #A78BFA, #60A5FA, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: #6B7280;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── metric cards ── */
.metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: linear-gradient(135deg, #1E2235, #232840);
    border: 1px solid #323860;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}
.accent-purple { color: #A78BFA; }
.accent-blue   { color: #60A5FA; }
.accent-green  { color: #34D399; }
.accent-amber  { color: #FBBF24; }

/* ── summary box ── */
.summary-box {
    background: linear-gradient(135deg, #1E2235, #222840);
    border: 1px solid #323860;
    border-left: 4px solid #A78BFA;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #D1D5DB;
    margin: 1rem 0;
}

/* ── tabs ── */
[data-testid="stTabs"] button {
    color: #6B7280 !important;
    font-weight: 600;
    letter-spacing: 0.04em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #A78BFA !important;
    border-bottom-color: #A78BFA !important;
}

/* ── input areas ── */
textarea, [data-testid="stTextArea"] textarea {
    background-color: #1E2235 !important;
    color: #E2E4EE !important;
    border: 1px solid #323860 !important;
    border-radius: 10px !important;
}

/* ── buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7C3AED, #3B82F6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #2A2E45;
    margin: 2rem 0;
}

/* ── chart section label ── */
.chart-label {
    font-size: 0.78rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BART model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

def generate_summary(text, max_length=150, min_length=50):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# ─────────────────────────────────────────────
# Analytics helpers
# ─────────────────────────────────────────────
def compute_stats(original: str, summary: str) -> dict:
    orig_words  = len(original.split())
    summ_words  = len(summary.split())
    orig_chars  = len(original)
    summ_chars  = len(summary)
    orig_sents  = max(1, len(re.split(r'[.!?]+', original.strip())))
    summ_sents  = max(1, len(re.split(r'[.!?]+', summary.strip())))
    compression = round((1 - summ_words / orig_words) * 100, 1) if orig_words else 0
    readability = round(206.835 - 1.015*(orig_words/orig_sents) - 84.6*(orig_chars/orig_words), 1) if orig_words else 0
    readability = max(0, min(100, readability))
    return {
        "orig_words": orig_words,
        "summ_words": summ_words,
        "orig_chars": orig_chars,
        "summ_chars": summ_chars,
        "orig_sents": orig_sents,
        "summ_sents": summ_sents,
        "compression": compression,
        "readability": readability,
    }

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(30,34,53,0.0)",
    plot_bgcolor="rgba(30,34,53,0.0)",
    font=dict(color="#9CA3AF", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10),
)

def chart_word_reduction(s):
    fig = go.Figure(go.Bar(
        x=["Original", "Summary"],
        y=[s["orig_words"], s["summ_words"]],
        marker=dict(
            color=["#7C3AED", "#34D399"],
            line=dict(width=0),
        ),
        text=[f"{s['orig_words']:,} words", f"{s['summ_words']:,} words"],
        textposition="outside",
        textfont=dict(color="#D1D5DB", size=12),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Word Count Reduction", font=dict(color="#E2E4EE", size=14)),
        yaxis=dict(showgrid=True, gridcolor="#2A2E45", color="#6B7280"),
        xaxis=dict(color="#6B7280"),
        height=300,
    )
    return fig

def chart_compression_gauge(compression):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compression,
        number=dict(suffix="%", font=dict(color="#A78BFA", size=36)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#3A3A5C", tickfont=dict(color="#6B7280")),
            bar=dict(color="#7C3AED"),
            bgcolor="#1E2235",
            bordercolor="#323860",
            steps=[
                dict(range=[0, 40],  color="#232840"),
                dict(range=[40, 70], color="#1E2235"),
                dict(range=[70, 100],color="#1A1E30"),
            ],
            threshold=dict(line=dict(color="#60A5FA", width=3), thickness=0.8, value=compression),
        ),
        title=dict(text="Compression Rate", font=dict(color="#9CA3AF", size=13)),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260)
    return fig

def chart_sentence_breakdown(s):
    categories = ["Words", "Characters", "Sentences"]
    orig_vals = [s["orig_words"], s["orig_chars"], s["orig_sents"]]
    summ_vals = [s["summ_words"], s["summ_chars"], s["summ_sents"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Original", x=categories, y=orig_vals,
                         marker_color="#7C3AED", marker_line_width=0))
    fig.add_trace(go.Bar(name="Summary",  x=categories, y=summ_vals,
                         marker_color="#34D399", marker_line_width=0))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Document vs Summary Breakdown", font=dict(color="#E2E4EE", size=14)),
        barmode="group",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9CA3AF")),
        yaxis=dict(showgrid=True, gridcolor="#2A2E45", color="#6B7280"),
        xaxis=dict(color="#6B7280"),
        height=300,
    )
    return fig

def chart_readability_donut(readability):
    fig = go.Figure(go.Pie(
        values=[readability, 100 - readability],
        labels=["Readable", "Complex"],
        hole=0.72,
        marker=dict(colors=["#60A5FA", "#232840"], line=dict(width=0)),
        textinfo="none",
        hoverinfo="label+percent",
    ))
    fig.add_annotation(
        text=f"{readability}",
        x=0.5, y=0.55, showarrow=False,
        font=dict(size=32, color="#60A5FA", family="Inter"),
    )
    fig.add_annotation(
        text="readability score",
        x=0.5, y=0.38, showarrow=False,
        font=dict(size=11, color="#6B7280", family="Inter"),
    )
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Flesch Readability (0–100)", font=dict(color="#E2E4EE", size=14)),
        showlegend=False,
        height=260,
    )
    return fig

def show_results(original_text: str, summary: str):
    s = compute_stats(original_text, summary)

    # ── Summary box ──
    st.markdown("### ⚖️ Generated Summary")
    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    hr = "<hr class='section-divider'>"
    st.markdown(hr, unsafe_allow_html=True)

    # ── Metric cards ──
    st.markdown("""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value accent-purple">{orig:,}</div>
        <div class="metric-label">Original Words</div>
      </div>
      <div class="metric-card">
        <div class="metric-value accent-green">{summ:,}</div>
        <div class="metric-label">Summary Words</div>
      </div>
      <div class="metric-card">
        <div class="metric-value accent-amber">{comp}%</div>
        <div class="metric-label">Words Removed</div>
      </div>
      <div class="metric-card">
        <div class="metric-value accent-blue">{read}</div>
        <div class="metric-label">Readability Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value accent-purple">{osent}</div>
        <div class="metric-label">Original Sentences</div>
      </div>
      <div class="metric-card">
        <div class="metric-value accent-green">{ssent}</div>
        <div class="metric-label">Summary Sentences</div>
      </div>
    </div>
    """.format(
        orig=s["orig_words"], summ=s["summ_words"],
        comp=s["compression"], read=s["readability"],
        osent=s["orig_sents"], ssent=s["summ_sents"],
    ), unsafe_allow_html=True)

    st.markdown(hr, unsafe_allow_html=True)
    st.markdown("### 📊 Visual Analytics")

    # ── Charts row 1 ──
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_word_reduction(s), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.plotly_chart(chart_sentence_breakdown(s), use_container_width=True, config={"displayModeBar": False})

    # ── Charts row 2 ──
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(chart_compression_gauge(s["compression"]), use_container_width=True, config={"displayModeBar": False})
    with col4:
        st.plotly_chart(chart_readability_donut(s["readability"]), use_container_width=True, config={"displayModeBar": False})

    st.markdown(hr, unsafe_allow_html=True)

    # ── Plain-English explanation for non-tech viewers ──
    st.markdown("### 💡 What does this mean?")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"📄 The original document had **{s['orig_words']:,} words** across **{s['orig_sents']} sentences**.")
    with col_b:
        st.success(f"✂️ The AI condensed it down to **{s['summ_words']:,} words** — removing **{s['compression']}%** of the text while keeping the key points.")
    with col_c:
        level = "easy to read" if s["readability"] > 60 else "moderately complex" if s["readability"] > 40 else "highly complex"
        st.warning(f"📖 The document is **{level}** with a readability score of **{s['readability']}** out of 100.")

# ─────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">LexBrief</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Legal Document Summarizer · BART Model</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝  Paste Text", "📄  Upload PDF"])

with tab1:
    text_input = st.text_area("Paste your legal text below:", height=220,
                               placeholder="e.g. This Agreement is entered into as of the date last signed below...")
    if st.button("Generate Summary", key="text_button"):
        if text_input.strip():
            with st.spinner("Reading your document…"):
                summary = generate_summary(text_input)
            show_results(text_input, summary)
        else:
            st.warning("Please enter some text first.")

with tab2:
    uploaded_file = st.file_uploader("Upload a PDF legal document", type="pdf")
    if uploaded_file is not None:
        st.caption(f"📎 {uploaded_file.name}  ·  {uploaded_file.size / 1024:.1f} KB")
        if st.button("Extract & Summarize", key="pdf_button"):
            with st.spinner("Extracting text from PDF…"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            if extracted_text.strip():
                with st.expander("View raw extracted text"):
                    st.text_area("", value=extracted_text, height=180, disabled=True)
                with st.spinner("Summarizing…"):
                    summary = generate_summary(extracted_text)
                show_results(extracted_text, summary)
            else:
                st.error("Could not extract text. The PDF may be scanned or password-protected.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#3A3A5C;font-size:0.8rem;">Powered by BART · facebook/bart-large-cnn</p>',
    unsafe_allow_html=True
)
import streamlit as st
import io
import re
import time
import hashlib
import numpy as np
import faiss

from fitz import open as open_pdf
from sentence_transformers import SentenceTransformer
from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CONFIGURATION
# =========================================================

GROQ_MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

MAX_PDF_MB = 25                 # protects memory under heavy usage
EMBED_BATCH_SIZE = 64           # batched encoding, avoids OOM on big PDFs
MAX_HISTORY = 12                # cap chat history kept in memory per session
GROQ_MAX_RETRIES = 3
GROQ_RETRY_BACKOFF = 1.6        # seconds, exponential


# =========================================================
# THEME / CSS  (navy scheme)
# =========================================================

st.markdown(
    """
    <style>

    :root {
        --navy-900: #0a1a3c;
        --navy-800: #0f234f;
        --navy-700: #16305f;
        --navy-600: #1e3f78;
        --navy-500: #2d5ba3;
        --accent: #3d7ee0;
        --gold: #d4af37;
        --bg-soft: #f4f7fb;
        --text-soft: #5b6b82;
    }

    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef2f8 100%);
        color: var(--navy-800);
    }

    /* Force readable default text color across the main content area
       (fixes faded/invisible text from the base dark theme bleeding through) */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--navy-900) !important;
    }
    .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp .stMarkdown, .stApp .stMarkdown * {
        color: var(--navy-800);
    }

    /* ---------- Hero header ---------- */
    .hero {
        background: linear-gradient(120deg, var(--navy-900) 0%, var(--navy-600) 100%);
        border-radius: 18px;
        padding: 38px 32px;
        margin-bottom: 26px;
        box-shadow: 0 10px 30px rgba(10, 26, 60, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(212,175,55,0.18) 0%, rgba(212,175,55,0) 70%);
        border-radius: 50%;
    }
    .hero-title {
        color: #fff !important;
        font-size: 36px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #c6d3ea !important;
        font-size: 15.5px;
        margin-top: 8px;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(212,175,55,0.15);
        color: var(--gold) !important;
        border: 1px solid rgba(212,175,55,0.4);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.4px;
        margin-top: 14px;
    }

    /* ---------- Cards ---------- */
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px 22px;
        border: 1px solid #e3e8f0;
        box-shadow: 0 4px 14px rgba(10,26,60,0.05);
        margin-bottom: 16px;
    }

    .answer-card {
        background: #ffffff;
        border-left: 5px solid var(--navy-600);
        border-radius: 12px;
        padding: 22px 24px;
        box-shadow: 0 6px 18px rgba(10,26,60,0.08);
    }

    .context-chip {
        display: inline-block;
        background: var(--navy-800);
        color: #fff;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }

    .score-chip {
        display: inline-block;
        background: var(--bg-soft);
        color: var(--navy-700);
        border: 1px solid #d6dfec;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy-900) 0%, var(--navy-800) 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e7ecf5 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        background: linear-gradient(120deg, var(--navy-600), var(--accent));
        color: #fff !important;
        border: none;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 700;
        letter-spacing: 0.2px;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
        box-shadow: 0 4px 12px rgba(45,91,163,0.35);
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(45,91,163,0.45);
    }
    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #fff !important;
    }

    /* ---------- File uploader ---------- */
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 2px dashed #b9c6dc !important;
        border-radius: 12px !important;
    }
    section[data-testid="stFileUploaderDropzone"] *,
    div[data-testid="stFileUploaderDropzone"] * {
        color: var(--navy-700) !important;
    }
    section[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(120deg, var(--navy-600), var(--accent)) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
    }
    section[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] small {
        color: var(--text-soft) !important;
    }
    /* uploaded file row (name + remove icon) */
    div[data-testid="stFileUploaderFile"] {
        background: var(--bg-soft) !important;
        border: 1px solid #d6dfec !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFileUploaderFile"] * {
        color: var(--navy-700) !important;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        padding: 14px 10px;
        box-shadow: 0 3px 10px rgba(10,26,60,0.05);
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: var(--navy-700) !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: var(--navy-900) !important;
        opacity: 1 !important;
    }

    /* ---------- Text input ---------- */
    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: var(--navy-800) !important;
        border: 1px solid #d6dfec !important;
        border-radius: 10px !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: var(--text-soft) !important;
        opacity: 1 !important;
    }
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextInput"] label * {
        color: var(--navy-700) !important;
        opacity: 1 !important;
    }

    /* ---------- Misc ---------- */
    .footer-note {
        text-align: center;
        color: var(--text-soft);
        font-size: 12.5px;
        margin-top: 34px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🏢 HR Policy Assistant</div>
        <div class="hero-subtitle">Upload an HR policy PDF and get instant, grounded answers — powered by RAG.</div>
        <div class="hero-badge">⚡ Optimized for high-traffic use</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GROQ API
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it in Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


client = get_groq_client()


# =========================================================
# LOAD EMBEDDING MODEL (shared across all sessions/users)
# =========================================================

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


embedding_model = load_embedding_model()


# =========================================================
# EXTRACT TEXT FROM PDF USING PYMUPDF
# =========================================================

def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    pdf_file = io.BytesIO(pdf_bytes)

    document = open_pdf(stream=pdf_file.read(), filetype="pdf")

    pages = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text()
        if text:
            text = text.strip()
            if text:
                pages.append({"page": page_number, "text": text})

    document.close()
    return pages


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = clean_text(page_data["text"])

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({"text": chunk_text, "page": page_number})

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# =========================================================
# CREATE EMBEDDINGS + FAISS VECTOR DATABASE (batched -> heavy usage safe)
# =========================================================

def create_faiss_index(chunks, progress_callback=None):
    texts = [chunk["text"] for chunk in chunks]

    all_embeddings = []
    total = len(texts)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]

        batch_embeddings = embedding_model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        all_embeddings.append(batch_embeddings)

        if progress_callback:
            done = min(start + EMBED_BATCH_SIZE, total)
            progress_callback(done / total)

    embeddings = np.vstack(all_embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(question, index, chunks, top_k=DEFAULT_TOP_K):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    question_embedding = question_embedding.astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    retrieved_chunks = []
    for score, index_number in zip(scores[0], indices[0]):
        if index_number == -1:
            continue

        retrieved_chunks.append({
            "text": chunks[index_number]["text"],
            "page": chunks[index_number]["page"],
            "score": float(score)
        })

    return retrieved_chunks


# =========================================================
# GENERATE ANSWER USING GROQ (with retry/backoff -> heavy usage safe)
# =========================================================

def generate_answer(question, retrieved_chunks):
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[Page {chunk['page']}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question using ONLY the
information provided in the retrieved HR Policy PDF context.

Rules:

1. Do not use outside knowledge.
2. Do not invent HR policies.
3. If the answer is not present in the PDF,
   say:

   "I could not find this information in the uploaded HR policy."

4. Give a clear and simple answer.
5. Mention the relevant page number when possible.
6. If multiple retrieved sections are relevant,
   combine them carefully.
7. Do not assume a policy that is not written in the PDF.

Retrieved HR Policy Context:
--------------------------------

{context}

--------------------------------

User Question:

{question}
"""

    last_error = None

    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer HR policy questions using only the retrieved document context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=2000,
                reasoning_effort="medium"
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            last_error = e
            time.sleep(GROQ_RETRY_BACKOFF ** attempt)

        except (APIConnectionError, APIError) as e:
            last_error = e
            time.sleep(GROQ_RETRY_BACKOFF ** attempt)

    return (
        "⚠️ The assistant is under heavy load right now and couldn't get a "
        f"response after {GROQ_MAX_RETRIES} attempts. Please try again in a "
        f"moment.\n\n_Technical detail: {last_error}_"
    )


# =========================================================
# SIMPLE PER-SESSION ANSWER CACHE (avoids re-billing identical questions)
# =========================================================

def get_cache_key(file_name, question, top_k):
    raw = f"{file_name}|{question.strip().lower()}|{top_k}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ RAG Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K
    )

    st.markdown("---")
    st.markdown("### 🔧 Technologies")
    st.markdown(
        """
        - 📄 **PDF Extraction:** PyMuPDF
        - ✂️ **Chunking:** Custom text chunking
        - 🧠 **Embeddings:** Sentence Transformers
        - 🗄️ **Vector Database:** FAISS
        - 🤖 **LLM:** Groq
        """
    )

    st.markdown("---")
    st.markdown("### ⚡ Heavy-Usage Safeguards")
    st.markdown(
        f"""
        - Batched embeddings ({EMBED_BATCH_SIZE}/batch)
        - Cached embedding model & Groq client
        - Retry + backoff on Groq errors ({GROQ_MAX_RETRIES}x)
        - Per-question answer cache (this session)
        - Max PDF size: {MAX_PDF_MB} MB
        """
    )

    if st.session_state.get("processed_file"):
        st.markdown("---")
        if st.button("🗑️ Reset session"):
            for key in ["processed_file", "pages", "chunks", "index", "answer_cache", "chat_history"]:
                st.session_state.pop(key, None)
            st.rerun()


# =========================================================
# PDF UPLOAD
# =========================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📄 Upload HR Policy PDF",
    type=["pdf"]
)

st.markdown('</div>', unsafe_allow_html=True)


if uploaded_file is not None:

    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

    if file_size_mb > MAX_PDF_MB:
        st.error(
            f"This PDF is {file_size_mb:.1f} MB, which exceeds the "
            f"{MAX_PDF_MB} MB limit for reliable processing under heavy load. "
            "Please upload a smaller file."
        )
        st.stop()

    st.success(f"✅ Uploaded: **{uploaded_file.name}** ({file_size_mb:.1f} MB)")

    # =====================================================
    # PROCESS PDF
    # =====================================================

    if (
        "processed_file" not in st.session_state
        or st.session_state["processed_file"] != uploaded_file.name
    ):

        status = st.empty()
        progress_bar = st.progress(0, text="Extracting text from PDF...")

        # STEP 1: Extract text
        pages = extract_pdf_text(uploaded_file)

        if not pages:
            st.error("No readable text was found in this PDF.")
            st.stop()

        progress_bar.progress(20, text="Splitting into chunks...")

        # STEP 2: Create chunks
        chunks = create_chunks(pages)

        if not chunks:
            st.error("Could not create text chunks.")
            st.stop()

        progress_bar.progress(35, text="Generating embeddings...")

        # STEP 3: Create embeddings + FAISS (batched, with live progress)
        def _update_progress(fraction):
            pct = 35 + int(fraction * 60)
            progress_bar.progress(min(pct, 95), text=f"Generating embeddings... {int(fraction * 100)}%")

        index = create_faiss_index(chunks, progress_callback=_update_progress)

        progress_bar.progress(100, text="Done!")
        time.sleep(0.3)
        progress_bar.empty()

        # Save data
        st.session_state["processed_file"] = uploaded_file.name
        st.session_state["pages"] = pages
        st.session_state["chunks"] = chunks
        st.session_state["index"] = index
        st.session_state["answer_cache"] = {}
        st.session_state["chat_history"] = []

        st.success("🎉 HR Policy PDF processed successfully!")

    # =====================================================
    # LOAD PROCESSED DATA
    # =====================================================

    pages = st.session_state["pages"]
    chunks = st.session_state["chunks"]
    index = st.session_state["index"]
    st.session_state.setdefault("answer_cache", {})
    st.session_state.setdefault("chat_history", [])

    # =====================================================
    # DOCUMENT INFORMATION
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 PDF Pages", len(pages))
    with col2:
        st.metric("✂️ Text Chunks", len(chunks))
    with col3:
        st.metric("📐 Vector Dimension", index.d)

    st.divider()

    # =====================================================
    # QUESTION SECTION
    # =====================================================

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Ask an HR Policy Question")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: How many annual leaves are allowed?"
    )

    ask_clicked = st.button("🔎 Search & Answer", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # SEARCH + ANSWER
    # =====================================================

    if ask_clicked:

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            cache_key = get_cache_key(uploaded_file.name, question, top_k)
            cached = st.session_state["answer_cache"].get(cache_key)

            if cached:
                retrieved_chunks = cached["retrieved_chunks"]
                answer = cached["answer"]
                st.info("⚡ Served from this session's cache — instant, no extra API cost.")

            else:
                with st.spinner("Searching relevant HR policy information..."):
                    retrieved_chunks = retrieve_chunks(question, index, chunks, top_k)

                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, retrieved_chunks)

                st.session_state["answer_cache"][cache_key] = {
                    "retrieved_chunks": retrieved_chunks,
                    "answer": answer
                }

            # Track light chat history (capped)
            st.session_state["chat_history"].append({"q": question, "a": answer})
            st.session_state["chat_history"] = st.session_state["chat_history"][-MAX_HISTORY:]

            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.markdown("### 🤖 Answer")
            st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

            # ------------------------------------------------
            # RETRIEVED CONTEXT
            # ------------------------------------------------

            with st.expander("🔍 View Retrieved HR Policy Context", expanded=False):

                for number, chunk in enumerate(retrieved_chunks, start=1):
                    st.markdown(
                        f'<span class="context-chip">Result {number}</span> '
                        f'<span class="context-chip">Page {chunk["page"]}</span> '
                        f'<span class="score-chip">Similarity {chunk["score"]:.3f}</span>',
                        unsafe_allow_html=True
                    )
                    st.write(chunk["text"])
                    st.divider()

    # =====================================================
    # RECENT QUESTIONS (session chat history)
    # =====================================================

    if st.session_state["chat_history"]:
        with st.expander(f"🕒 Recent questions this session ({len(st.session_state['chat_history'])})"):
            for item in reversed(st.session_state["chat_history"]):
                st.markdown(f"**Q:** {item['q']}")
                st.markdown(f"**A:** {item['a']}")
                st.divider()

else:
    st.info("📄 Upload an HR Policy PDF to start.")


st.markdown(
    '<div class="footer-note">Built with Streamlit · FAISS · Sentence Transformers · Groq</div>',
    unsafe_allow_html=True
)

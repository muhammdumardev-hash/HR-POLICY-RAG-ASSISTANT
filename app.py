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

GROQ_MODEL = st.secrets.get(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

DEFAULT_TOP_K = 5

MAX_PDF_MB = 25
EMBED_BATCH_SIZE = 64

MAX_HISTORY = 12

GROQ_MAX_RETRIES = 3
GROQ_RETRY_BACKOFF = 1.6


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN BACKGROUND
       ===================================================== */

    .stApp {
        background: linear-gradient(
            180deg,
            #f7f9fc 0%,
            #eef2f8 100%
        );
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        background: linear-gradient(
            120deg,
            #0a1a3c 0%,
            #1e3f78 100%
        );

        border-radius: 18px;

        padding: 34px 32px;

        margin-bottom: 25px;

        box-shadow:
            0 10px 30px rgba(10, 26, 60, 0.20);

        position: relative;

        overflow: hidden;
    }

    .hero-title {
        color: white;

        font-size: 36px;

        font-weight: 800;

        margin: 0;
    }

    .hero-subtitle {
        color: #d6e0f0;

        font-size: 15px;

        margin-top: 8px;
    }

    .hero-badge {
        display: inline-block;

        margin-top: 15px;

        padding: 5px 12px;

        border-radius: 20px;

        background: rgba(255,255,255,0.12);

        border: 1px solid rgba(255,255,255,0.20);

        color: #ffffff;

        font-size: 12px;

        font-weight: 600;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    div[data-testid="stFileUploader"] {
        background: #ffffff;

        border: 2px dashed #2d5ba3;

        border-radius: 14px;

        padding: 12px;

        box-shadow:
            0 4px 14px rgba(10,26,60,0.06);
    }

    div[data-testid="stFileUploader"] section {
        background: #ffffff !important;

        border-radius: 10px;
    }

    div[data-testid="stFileUploader"] button {
        background: #2d5ba3 !important;

        color: white !important;

        border: none !important;

        border-radius: 8px !important;

        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] small {
        color: #5b6b82 !important;
    }


    /* =====================================================
       QUESTION INPUT
       ===================================================== */

    div[data-testid="stTextInput"] input {
        background: #ffffff !important;

        color: #0a1a3c !important;

        border: 1px solid #cfd8e6 !important;

        border-radius: 10px !important;

        padding: 12px !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border: 2px solid #2d5ba3 !important;

        box-shadow:
            0 0 0 2px rgba(45,91,163,0.10) !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    div.stButton > button {
        background: linear-gradient(
            120deg,
            #16305f,
            #2d5ba3
        );

        color: white;

        border: none;

        border-radius: 10px;

        padding: 10px 18px;

        font-weight: 700;

        box-shadow:
            0 4px 12px rgba(45,91,163,0.25);
    }

    div.stButton > button:hover {
        background: linear-gradient(
            120deg,
            #1e3f78,
            #3d7ee0
        );

        color: white;
    }


    /* =====================================================
       METRICS
       ===================================================== */

    div[data-testid="stMetric"] {
        background: #ffffff;

        border: 1px solid #e1e7f0;

        border-radius: 12px;

        padding: 14px;

        box-shadow:
            0 3px 10px rgba(10,26,60,0.05);
    }


    /* =====================================================
       ANSWER CARD
       ===================================================== */

    .answer-card {
        background: #ffffff;

        border-left: 5px solid #1e3f78;

        border-radius: 12px;

        padding: 20px 22px;

        color: #0a1a3c;

        box-shadow:
            0 6px 18px rgba(10,26,60,0.08);

        line-height: 1.7;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0a1a3c 0%,
            #0f234f 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: #e7ecf5 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }


    /* =====================================================
       CONTEXT CHIPS
       ===================================================== */

    .context-chip {
        display: inline-block;

        background: #16305f;

        color: white;

        padding: 4px 10px;

        border-radius: 6px;

        font-size: 12px;

        font-weight: 600;

        margin-right: 6px;
    }

    .score-chip {
        display: inline-block;

        background: #eef2f8;

        color: #16305f;

        border: 1px solid #d6dfec;

        padding: 4px 10px;

        border-radius: 6px;

        font-size: 12px;

        font-weight: 600;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-note {
        text-align: center;

        color: #6b7a90;

        font-size: 12px;

        margin-top: 35px;

        margin-bottom: 15px;
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

        <div class="hero-title">
            🏢 HR Policy Assistant
        </div>

        <div class="hero-subtitle">
            Upload an HR policy PDF and get instant,
            document-based answers using RAG.
        </div>

        <div class="hero-badge">
            ⚡ AI-Powered Document Assistant
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GROQ API
# =========================================================

try:

    GROQ_API_KEY = st.secrets[
        "GROQ_API_KEY"
    ]

except Exception:

    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it in Streamlit Cloud → Settings → Secrets."
    )

    st.stop()


@st.cache_resource
def get_groq_client():

    return Groq(
        api_key=GROQ_API_KEY
    )


client = get_groq_client()


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model


embedding_model = load_embedding_model()


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_pdf_text(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    pdf_file = io.BytesIO(
        pdf_bytes
    )

    document = open_pdf(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text()

        if text:

            text = text.strip()

            if text:

                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )

    document.close()

    return pages


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = clean_text(
            page_data["text"]
        )

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            if end >= len(text):

                break

            start = end - overlap

    return chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(
    chunks,
    progress_callback=None
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    all_embeddings = []

    total = len(texts)

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE
    ):

        batch = texts[
            start:start + EMBED_BATCH_SIZE
        ]

        batch_embeddings = embedding_model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        all_embeddings.append(
            batch_embeddings
        )

        if progress_callback:

            done = min(
                start + EMBED_BATCH_SIZE,
                total
            )

            progress_callback(
                done / total
            )

    embeddings = np.vstack(
        all_embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    top_k=DEFAULT_TOP_K
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    k = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        k
    )

    retrieved_chunks = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:

            continue

        retrieved_chunks.append(
            {
                "text":
                    chunks[index_number]["text"],

                "page":
                    chunks[index_number]["page"],

                "score":
                    float(score)
            }
        )

    return retrieved_chunks


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    retrieved_chunks
):

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question using ONLY
the information provided in the retrieved
HR Policy PDF context.

Rules:

1. Do not use outside knowledge.
2. Do not invent HR policies.
3. If the answer is not present in the PDF,
say:

"I could not find this information in the uploaded HR policy."

4. Give a clear and simple answer.
5. Mention the relevant page number when possible.
6. Combine multiple relevant sections carefully.
7. Never assume a policy that is not written in the PDF.

Retrieved HR Policy Context:
--------------------------------

{context}

--------------------------------

User Question:

{question}
"""

    last_error = None

    for attempt in range(
        1,
        GROQ_MAX_RETRIES + 1
    ):

        try:

            response = client.chat.completions.create(

                model=GROQ_MODEL,

                messages=[

                    {
                        "role": "system",
                        "content":
                        "You answer HR policy questions using only the retrieved document context."
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

            return response.choices[
                0
            ].message.content


        except RateLimitError as e:

            last_error = e

            time.sleep(
                GROQ_RETRY_BACKOFF ** attempt
            )


        except (
            APIConnectionError,
            APIError
        ) as e:

            last_error = e

            time.sleep(
                GROQ_RETRY_BACKOFF ** attempt
            )


    return (
        "⚠️ The assistant is temporarily "
        "under heavy load. Please try again "
        "in a moment."
    )


# =========================================================
# CACHE KEY
# =========================================================

def get_cache_key(
    file_name,
    question,
    top_k
):

    raw = (
        f"{file_name}|"
        f"{question.strip().lower()}|"
        f"{top_k}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ RAG Settings"
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K
    )

    st.markdown("---")

    st.markdown(
        "### 🔧 Technologies"
    )

    st.markdown(
        """
        📄 **PDF:** PyMuPDF

        ✂️ **Chunking:** Custom

        🧠 **Embeddings:** Sentence Transformers

        🗄️ **Vector DB:** FAISS

        🤖 **LLM:** Groq
        """
    )

    st.markdown("---")

    st.markdown(
        "### ⚡ Performance"
    )

    st.markdown(
        f"""
        • Batched embeddings: {EMBED_BATCH_SIZE}

        • Cached AI resources

        • API retry protection

        • Session answer cache

        • PDF limit: {MAX_PDF_MB} MB
        """
    )

    if st.session_state.get(
        "processed_file"
    ):

        st.markdown("---")

        if st.button(
            "🗑️ Reset Session"
        ):

            for key in [
                "processed_file",
                "pages",
                "chunks",
                "index",
                "answer_cache",
                "chat_history"
            ]:

                st.session_state.pop(
                    key,
                    None
                )

            st.rerun()


# =========================================================
# PDF UPLOAD
# =========================================================

st.subheader(
    "📄 Upload HR Policy PDF"
)

uploaded_file = st.file_uploader(
    "Choose an HR Policy PDF",
    type=["pdf"],
    help="Upload a text-based HR policy PDF."
)


# =========================================================
# IF PDF EXISTS
# =========================================================

if uploaded_file is not None:

    file_size_mb = (
        len(
            uploaded_file.getvalue()
        )
        /
        (1024 * 1024)
    )

    if file_size_mb > MAX_PDF_MB:

        st.error(
            f"PDF size is {file_size_mb:.1f} MB. "
            f"Maximum allowed size is {MAX_PDF_MB} MB."
        )

        st.stop()


    st.success(
        f"✅ {uploaded_file.name} uploaded successfully"
    )


    # =====================================================
    # PROCESS PDF
    # =====================================================

    if (
        "processed_file"
        not in st.session_state

        or

        st.session_state[
            "processed_file"
        ]
        != uploaded_file.name
    ):

        progress_bar = st.progress(
            0,
            text="Extracting PDF text..."
        )


        # -------------------------------------------------
        # STEP 1
        # -------------------------------------------------

        pages = extract_pdf_text(
            uploaded_file
        )

        if not pages:

            st.error(
                "No readable text was found in this PDF."
            )

            st.stop()


        progress_bar.progress(
            20,
            text="Creating text chunks..."
        )


        # -------------------------------------------------
        # STEP 2
        # -------------------------------------------------

        chunks = create_chunks(
            pages
        )

        if not chunks:

            st.error(
                "Could not create text chunks."
            )

            st.stop()


        progress_bar.progress(
            35,
            text="Generating embeddings..."
        )


        # -------------------------------------------------
        # STEP 3
        # -------------------------------------------------

        def update_progress(
            fraction
        ):

            percent = 35 + int(
                fraction * 60
            )

            progress_bar.progress(
                min(percent, 95),
                text=
                f"Generating embeddings... "
                f"{int(fraction * 100)}%"
            )


        index = create_faiss_index(
            chunks,
            progress_callback=
            update_progress
        )


        progress_bar.progress(
            100,
            text="PDF processing complete!"
        )

        time.sleep(
            0.3
        )

        progress_bar.empty()


        # -------------------------------------------------
        # SAVE DATA
        # -------------------------------------------------

        st.session_state[
            "processed_file"
        ] = uploaded_file.name

        st.session_state[
            "pages"
        ] = pages

        st.session_state[
            "chunks"
        ] = chunks

        st.session_state[
            "index"
        ] = index

        st.session_state[
            "answer_cache"
        ] = {}

        st.session_state[
            "chat_history"
        ] = []


        st.success(
            "🎉 HR Policy PDF processed successfully!"
        )


    # =====================================================
    # LOAD DATA
    # =====================================================

    pages = st.session_state[
        "pages"
    ]

    chunks = st.session_state[
        "chunks"
    ]

    index = st.session_state[
        "index"
    ]


    st.session_state.setdefault(
        "answer_cache",
        {}
    )

    st.session_state.setdefault(
        "chat_history",
        []
    )


    # =====================================================
    # DOCUMENT INFORMATION
    # =====================================================

    st.markdown(
        "### 📊 Document Information"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📄 PDF Pages",
            len(pages)
        )


    with col2:

        st.metric(
            "✂️ Text Chunks",
            len(chunks)
        )


    with col3:

        st.metric(
            "📐 Vector Dimension",
            index.d
        )


    st.divider()


    # =====================================================
    # QUESTION
    # =====================================================

    st.subheader(
        "💬 Ask an HR Policy Question"
    )

    question = st.text_input(
        "Enter your question",
        placeholder=
        "Example: How many annual leaves are allowed?"
    )


    ask_clicked = st.button(
        "🔎 Search & Answer",
        type="primary"
    )


    # =====================================================
    # ANSWER
    # =====================================================

    if ask_clicked:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )


        else:

            cache_key = get_cache_key(
                uploaded_file.name,
                question,
                top_k
            )


            cached = st.session_state[
                "answer_cache"
            ].get(
                cache_key
            )


            if cached:

                retrieved_chunks = cached[
                    "retrieved_chunks"
                ]

                answer = cached[
                    "answer"
                ]

                st.info(
                    "⚡ Answer loaded from session cache."
                )


            else:

                with st.spinner(
                    "🔍 Searching HR policy..."
                ):

                    retrieved_chunks = retrieve_chunks(
                        question,
                        index,
                        chunks,
                        top_k
                    )


                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        retrieved_chunks
                    )


                st.session_state[
                    "answer_cache"
                ][cache_key] = {

                    "retrieved_chunks":
                        retrieved_chunks,

                    "answer":
                        answer
                }


            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            st.session_state[
                "chat_history"
            ].append(
                {
                    "q": question,
                    "a": answer
                }
            )


            st.session_state[
                "chat_history"
            ] = st.session_state[
                "chat_history"
            ][-MAX_HISTORY:]


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.markdown(
                "### 🤖 Answer"
            )

            st.markdown(
                f"""
                <div class="answer-card">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # RETRIEVED CONTEXT
            # ------------------------------------------------

            with st.expander(
                "🔍 View Retrieved HR Policy Context"
            ):

                for number, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.markdown(
                        f"""
                        <span class="context-chip">
                            Result {number}
                        </span>

                        <span class="context-chip">
                            Page {chunk['page']}
                        </span>

                        <span class="score-chip">
                            Similarity {chunk['score']:.3f}
                        </span>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.divider()


    # =====================================================
    # RECENT QUESTIONS
    # =====================================================

    if st.session_state[
        "chat_history"
    ]:

        with st.expander(
            f"🕒 Recent Questions "
            f"({len(st.session_state['chat_history'])})"
        ):

            for item in reversed(
                st.session_state[
                    "chat_history"
                ]
            ):

                st.markdown(
                    f"**Q:** {item['q']}"
                )

                st.markdown(
                    f"**A:** {item['a']}"
                )

                st.divider()


else:

    st.info(
        "📄 Upload an HR Policy PDF above to start."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-note">
        Built with Streamlit · FAISS · Sentence Transformers · Groq
    </div>
    """,
    unsafe_allow_html=True
)

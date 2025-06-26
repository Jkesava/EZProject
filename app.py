import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

# Optional: for Windows users
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="GenAI Assistant", layout="wide")
st.title("📘 Smart Assistant for Research Summarization")

# ========== TEXT EXTRACTION ==========

def extract_text_from_pdf(file):
    file_bytes = BytesIO(file.read())
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.strip()

def extract_text_via_ocr(file):
    st.info("🔄 Performing OCR on image-based PDF...")
    images = convert_from_bytes(file.read())
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text.strip()

# ========== LOAD MODELS ==========

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embedder, qa

embedder, qa_pipeline = load_models()

# ========== ADVANCED Q&A FUNCTION ==========

def get_best_answer(text, question):
    # Section-aware chunking
    headings = ["Objective", "Problem Statement", "Functional Requirements",
                "Contextual Understanding", "Auto Summary", "Interaction Modes",
                "Application Architecture", "Bonus Features", "Submission", "Evaluation"]

    sections = []
    current = ""
    for line in text.split('\n'):
        if any(h.lower() in line.lower() for h in headings):
            if current:
                sections.append(current.strip())
                current = ""
        current += line + " "
    if current:
        sections.append(current.strip())

    # Embedding-based search
    embeddings = embedder.encode(sections)
    q_embed = embedder.encode([question])[0]
    similarities = cosine_similarity([q_embed], embeddings)[0]
    best_idx = np.argmax(similarities)
    best_chunk = sections[best_idx]

    result = qa_pipeline(question=question, context=best_chunk)
    answer = result.get("answer", "").strip()
    score = result.get("score", 0.0)

    if not answer:
        answer = "❓ Sorry, I couldn't find an answer in the document."

    return answer, best_chunk, score

# ========== STREAMLIT LOGIC ==========

uploaded_file = st.file_uploader("📄 Upload a PDF or TXT file", type=["pdf", "txt"])
text = ""

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    if uploaded_file.type == "application/pdf":
        uploaded_file.seek(0)
        text = extract_text_from_pdf(uploaded_file)
        if len(text.strip()) < 20:
            uploaded_file.seek(0)
            text = extract_text_via_ocr(uploaded_file)

    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    else:
        text = "❌ Unsupported file format."

    st.subheader("📄 Document Preview:")
    st.write(text[:1500] if text else "⚠️ No readable content found.")

    # ========== ASK ANYTHING ==========
    st.markdown("---")
    st.subheader("💬 Ask Anything (Q&A)")

    user_question = st.text_input("🔎 Ask a question based on the document:")

    if user_question and text:
        st.info("🤖 Processing your question...")
        answer, justification, score = get_best_answer(text, user_question)

        st.success(f"**Answer:** {answer}")
        st.caption(f"📌 Justified by: \"{justification[:300]}...\"")
        st.caption(f"🔍 Confidence Score: {score:.2f}")

else:
    st.info("⬆️ Upload a file to begin.")

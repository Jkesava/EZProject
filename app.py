import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np

st.set_page_config(page_title="GenAI Assistant", layout="wide")
st.title("📘 Smart Assistant for Research Summarization")

# ========== TEXT EXTRACTION (PDF Only) ==========

def extract_text_from_pdf(file):
    file_bytes = BytesIO(file.read())
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.strip()

# ========== LOAD MODELS ==========

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embedder, qa

embedder, qa_pipeline = load_models()

# ========== SECTION-AWARE Q&A ==========

def get_best_answer(text, question):
    headings = ["Objective", "Problem Statement", "Functional Requirements",
                "Contextual Understanding", "Auto Summary", "Interaction Modes",
                "Bonus Features", "Submission", "Evaluation"]

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

# ========== STREAMLIT UI ==========

uploaded_file = st.file_uploader("📄 Upload a PDF or TXT file", type=["pdf", "txt"])
text = ""

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    if uploaded_file.type == "application/pdf":
        uploaded_file.seek(0)
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 Document Preview:")
    st.write(text[:1500] if text else "⚠️ No readable content found.")

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

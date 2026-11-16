import zipfile
import os
import streamlit as st
import torch
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = "models/siamese_assignment_model"
THRESHOLD = 0.20

st.set_page_config(
    page_title="Duplicate Assignment Detection",
    page_icon="📚",
    layout="centered"
)

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_DIR):
        os.makedirs("models", exist_ok=True)

        with zipfile.ZipFile("siamese_model.zip", "r") as zip_ref:
            zip_ref.extractall("models")

    return SentenceTransformer(MODEL_DIR)

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_chunks(text, chunk_size=180, overlap=30):
    words = text.split()
    chunks = []

    if not words:
        return chunks

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

def compare_documents(model, text1, text2):
    chunks1 = split_into_chunks(text1)
    chunks2 = split_into_chunks(text2)

    if not chunks1 or not chunks2:
        raise ValueError("One of the PDFs contains no extractable text.")

    embeddings1 = model.encode(
        chunks1,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings2 = model.encode(
        chunks2,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(embeddings1, embeddings2)

    score = float(similarities.max())
    score = max(0.0, min(1.0, score))

    return score

st.title("📚 Duplicate Assignment Detection")
st.subheader("Using Deep Learning")

st.write(
    "Upload two assignment PDFs to check their semantic similarity "
    "using a trained Siamese Sentence Transformer model."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    pdf1 = st.file_uploader(
        "Upload Assignment 1",
        type=["pdf"],
        key="pdf1"
    )

with col2:
    pdf2 = st.file_uploader(
        "Upload Assignment 2",
        type=["pdf"],
        key="pdf2"
    )

if st.button("Compare Assignments", type="primary", use_container_width=True):

    if pdf1 is None or pdf2 is None:
        st.warning("Please upload both PDF files.")
    else:
        try:
            with st.spinner("Loading model and comparing assignments..."):

                model = load_model()

                text1 = clean_text(extract_pdf(pdf1))
                text2 = clean_text(extract_pdf(pdf2))

                if len(text1.split()) < 20 or len(text2.split()) < 20:
                    raise ValueError(
                        "Both PDFs must contain at least 20 readable words."
                    )

                score = compare_documents(model, text1, text2)

            st.divider()
            st.subheader("Comparison Results")

            st.metric(
                "Similarity Percentage",
                f"{score * 100:.2f}%"
            )

            st.progress(score)

            st.write(f"Similarity Score: **{score:.4f}**")
            st.write(f"Decision Threshold: **{THRESHOLD}**")

            if score >= THRESHOLD:
                st.error("RESULT: POTENTIALLY SIMILAR")
                st.write("Both assignments should be reviewed.")
            else:
                st.success("RESULT: LESS SIMILAR")
                st.write("No strong similarity was detected.")

            st.info(
                "This result is an automated screening signal, "
                "not proof of plagiarism. Human review is required."
            )

        except Exception as e:
            st.error(f"Comparison failed: {e}")

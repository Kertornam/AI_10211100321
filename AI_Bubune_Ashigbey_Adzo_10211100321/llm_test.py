import streamlit as st
import pandas as pd
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def run_llm_module_():
    st.title("🔍 LLM Q&A: Ghana Election Results with RAG + Gemini")
    
    # Step 1: API Key Input
    api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")
    if not api_key:
        st.warning("Please enter your Gemini API key.")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")

    # Step 2: Upload the dataset
    uploaded_file = st.file_uploader("📁 Upload Ghana_Election_Result.csv", type=["csv"])
    if uploaded_file is None:
        st.info("Upload the required dataset to continue.")
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Step 3: Embed the rows for retrieval
    st.text("🔄 Processing embeddings...")
    text_rows = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(text_rows)

    # Step 4: User Question Input
    question = st.text_input("❓ Ask a question about the election results:")
    if not question:
        return

    # Step 5: Embed the question
    query_embedding = embedder.encode([question])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Step 6: Select top-k relevant rows
    top_k = 5
    top_indices = similarities.argsort()[-top_k:][::-1]
    context = "\n".join([text_rows[i] for i in top_indices])

    # Step 7: Build Prompt and Call Gemini
    prompt = f"""
You are an AI assistant analyzing Ghana's election results.
Use the following dataset context to answer the user's question.
Only use the information provided — do not make things up.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}
Answer:"""

    st.subheader("🧠 Gemini Response")
    try:
        response = model.generate_content(prompt)
        st.success(response.text)
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")


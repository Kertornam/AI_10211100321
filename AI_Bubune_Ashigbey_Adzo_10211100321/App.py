import streamlit as st
from regression import run_regression
from clustering import run_clustering
from neural_network1 import run_neural_network
from llm_module_1 import run_llm_module  

st.set_page_config(page_title="AI Project Dashboard", layout="centered")

# Sidebar navigation
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio("Go to:", ["Home", "Regression", "Clustering", "Neural Network", "LLM"])

# Home Page (Personalized)
if section == "Home":
    st.title("🤖 AI Project Dashboard")
    st.markdown("""
    ### 🧠 Author: Bubune Adzo Ashigbey  
    #### 📚 Course: Introduction to Artificial Intelligence  
    #### 🏫 Academic City University College  
    ---

    This application demonstrates four core concepts in Artificial Intelligence through interactive mini-modules:

    1. 🔢 **Regression** – Predict continuous values using Linear Regression (California housing data)
    2. 🧩 **Clustering** – Explore unsupervised grouping of data using K-Means (customer segmentation)
    3. 🧠 **Neural Networks** – Train a simple classifier using MLP (scikit-learn)
    4. 💬 **LLM + RAG** – Ask natural language questions over election data using Gemini API + sentence embeddings

    ---

    ### 🛠️ Methodology (Overview)
    Each module follows a similar process:
    - Upload data or use a built-in dataset
    - Select features and settings interactively
    - Train and evaluate an AI model
    - View results visually or get predictions

    The LLM module uses **Retrieval-Augmented Generation (RAG)**:
    - Vector embeddings from the dataset are matched to the user's query
    - Relevant context is passed to Google's Gemini Pro model
    - Gemini generates an accurate, data-grounded response

    > ✅ Designed as part of a project-based AI exam to demonstrate core ML and NLP skills.

    ---
    """)

# Section logic
if section == "Regression":
    run_regression()
elif section == "Clustering":
    run_clustering()
elif section == "Neural Network":
    run_neural_network()
elif section == "LLM":
    run_llm_module()


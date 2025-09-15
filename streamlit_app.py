import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util


@st.cache_resource(show_spinner=False)
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    model = SentenceTransformer(os.path.join(base_dir, "sbert_model"))
    with open(os.path.join(base_dir, "qa_embeddings.pkl"), "rb") as f:
        question_embeddings = pickle.load(f)
    with open(os.path.join(base_dir, "qa_answers.pkl"), "rb") as f:
        answers = pickle.load(f)
    # Optional artifacts
    questions: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    q_path = os.path.join(base_dir, "qa_questions.pkl")
    c_path = os.path.join(base_dir, "qa_categories.pkl")
    if os.path.exists(q_path):
        with open(q_path, "rb") as f:
            questions = pickle.load(f)
    if os.path.exists(c_path):
        with open(c_path, "rb") as f:
            categories = pickle.load(f)
    return model, question_embeddings, answers, questions, categories


def retrieve_top_k(user_text: str, model, question_embeddings, answers: List[str], top_k: int) -> List[Tuple[str, float, int]]:
    user_embedding = model.encode([user_text], convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings).flatten()
    # Convert to numpy array to avoid tensor issues
    similarities = similarities.cpu().numpy()
    # Ensure top_k is at least 1
    top_k = max(1, top_k)
    top_idx = np.argsort(similarities)[::-1][:top_k]
    results: List[Tuple[str, float, int]] = []
    for idx in top_idx:
        results.append((answers[idx], float(similarities[idx]), int(idx)))
    return results


st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

st.title("💬 University FAQ Chatbot")
st.caption("SentenceTransformer-based FAQ Chatbot")


model, question_embeddings, answers, questions, categories = load_artifacts()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Match threshold", 0.0, 1.0, 0.35, 0.01)
    top_k = st.slider("Show top‑k matches", 1, 5, 3, 1)
    show_candidates = st.checkbox("Show candidates", value=False)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Type your message…")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Always use your SentenceTransformer retriever
    candidates = retrieve_top_k(prompt, model, question_embeddings, answers, top_k=top_k)

    # Build reply
    if not candidates:
        reply = "I'm not sure yet. Could you rephrase or ask something else?"
    else:
        best_answer, best_score, _ = candidates[0]
        if best_score < threshold:
            reply = "I'm not sure yet. Could you rephrase or ask something else?"
        else:
            reply = best_answer

    with st.chat_message("assistant"):
        st.markdown(reply)
        if show_candidates and candidates:
            with st.expander("View candidates"):
                for ans, score, idx in candidates:
                    meta = []
                    if questions and 0 <= idx < len(questions):
                        meta.append(f"Q: {questions[idx]}")
                    if categories and 0 <= idx < len(categories) and categories[idx]:
                        meta.append(f"Category: {categories[idx]}")
                    st.write(f"Score: {score:.3f} | idx: {idx}")
                    if meta:
                        st.caption(" | ".join(meta))
                    st.text(ans)
                    st.markdown("---")

    st.session_state.history.append(("assistant", reply))

st.markdown("\n\n")
st.caption("Tip: adjust the threshold if responses feel too generic or too strict.")
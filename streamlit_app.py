import os
import pickle
from typing import List, Tuple

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource(show_spinner=False)
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(base_dir, "qa_matrix.pkl"), "rb") as f:
        qa_matrix = pickle.load(f)
    with open(os.path.join(base_dir, "qa_answers.pkl"), "rb") as f:
        answers = pickle.load(f)
    return vectorizer, qa_matrix, answers


def retrieve_top_k(user_text: str, vectorizer, qa_matrix, answers: List[str], top_k: int) -> List[Tuple[str, float, int]]:
    user_vec = vectorizer.transform([user_text])
    sims = cosine_similarity(user_vec, qa_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results: List[Tuple[str, float, int]] = []
    for idx in top_idx:
        results.append((answers[idx], float(sims[idx]), int(idx)))
    return results


st.set_page_config(page_title="CSV Chatbot", page_icon="💬", layout="centered")

st.title("💬 Conversation CSV Chatbot")
st.caption("TF‑IDF retrieval over your Conversation.csv dataset")

vectorizer, qa_matrix, answers = load_artifacts()

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

    candidates = retrieve_top_k(prompt, vectorizer, qa_matrix, answers, top_k=top_k)
    best_answer, best_score, _ = candidates[0]
    if best_score < threshold:
        reply = "I’m not sure yet. Could you rephrase or ask something else?"
    else:
        reply = best_answer

    with st.chat_message("assistant"):
        st.markdown(reply)
        if show_candidates:
            with st.expander("View candidates"):
                for ans, score, idx in candidates:
                    st.write(f"Score: {score:.3f} | idx: {idx}")
                    st.text(ans)
                    st.markdown("---")

    st.session_state.history.append(("assistant", reply))

st.markdown("\n\n")
st.caption("Tip: adjust the threshold if responses feel too generic or too strict.")



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
st.caption("Multi-Algorithm Chatbot for Group Assignment - Compare different NLP approaches!")

# Group assignment instructions
with st.expander("📋 Instructions for Group Members"):
    st.markdown("""
    **How to implement your algorithm:**
    
    1. **Find your section** in the code (around line 115-125)
    2. **Replace the placeholder** with your implementation
    3. **Your function should return** a list of tuples: `[(answer, score, index), ...]`
    4. **Use the same interface** as Yong Zheng's SentenceTransformer example
    
    **Group Member Assignments:**
    - **Yong Zheng**: Deep Learning (SentenceTransformer) (✅ Completed)
    - **Ew Chiu Linn**: [Your algorithm choice - TF-IDF, BERT, RoBERTa, etc.]
    - **Chong Yee Yang**: [Your algorithm choice - TF-IDF, BERT, RoBERTa, etc.]
    
    **Testing your algorithm:**
    - Select your name from the dropdown
    - Test with various questions
    - Compare performance with other group members
    - Check the performance stats at the bottom
    """)

model, question_embeddings, answers, questions, categories = load_artifacts()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Match threshold", 0.0, 1.0, 0.35, 0.01)
    top_k = st.slider("Show top‑k matches", 1, 5, 3, 1)
    show_candidates = st.checkbox("Show candidates", value=False)
    
    # Group member algorithm selection
    st.subheader("Group Member Selection")
    algorithm_options = ["(All Members)", "Yong Zheng", "Ew Chiu Linn", "Chong Yee Yang"]
    selected_algorithm = st.selectbox("Choose group member's algorithm to test:", algorithm_options, index=0)
    active_algorithm = None if selected_algorithm == "(All Members)" else selected_algorithm

if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)
if "ratings" not in st.session_state:
    st.session_state.ratings = []  # list of dicts {turn, prompt, reply, score}
if "algorithm_stats" not in st.session_state:
    st.session_state.algorithm_stats = {}  # track algorithm usage and performance

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Type your message…")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show which group member's algorithm is being used
    if active_algorithm == "Yong Zheng" or active_algorithm is None:
        current_algorithm = "Yong Zheng (Deep Learning)"
    else:
        current_algorithm = active_algorithm
    with st.chat_message("assistant"):
        st.caption(f"🤖 Using: {current_algorithm}'s Algorithm")

    # Group member algorithm-specific processing
    if active_algorithm == "Yong Zheng" or active_algorithm is None:
        # Yong Zheng's SentenceTransformer implementation
        candidates = retrieve_top_k(prompt, model, question_embeddings, answers, top_k=top_k)
    
    elif active_algorithm == "Ew Chiu Linn":
        # Placeholder for Ew Chiu Linn's implementation
        st.warning("🚧 Ew Chiu Linn's algorithm implementation not yet available. Please implement this algorithm.")
        candidates = []
    
    elif active_algorithm == "Chong Yee Yang":
        # Placeholder for Chong Yee Yang's implementation
        st.warning("🚧 Chong Yee Yang's algorithm implementation not yet available. Please implement this algorithm.")
        candidates = []
    
    else:
        # Fallback to Yong Zheng's implementation
        candidates = retrieve_top_k(prompt, model, question_embeddings, answers, top_k=top_k)
    
    # Safety check: ensure we have at least one candidate
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
        if show_candidates:
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

    # Usability rating UI per assistant reply
    with st.container(border=True):
        st.write("Rate this response (1=poor, 5=excellent):")
        col1, col2 = st.columns([1, 3])
        with col1:
            rating = st.slider("Rating", 1, 5, 4, 1, key=f"rate_{len(st.session_state.history)}")
        with col2:
            feedback = st.text_input("Optional feedback", key=f"fb_{len(st.session_state.history)}")
        if st.button("Submit rating", key=f"submit_{len(st.session_state.history)}"):
            rating_data = {
                "turn": len(st.session_state.history),
                "prompt": prompt,
                "reply": reply,
                "score": int(rating),
                "feedback": feedback,
                "algorithm": current_algorithm,
                "best_score": best_score if candidates else 0.0
            }
            st.session_state.ratings.append(rating_data)
            
            # Track algorithm stats
            if current_algorithm not in st.session_state.algorithm_stats:
                st.session_state.algorithm_stats[current_algorithm] = {"count": 0, "total_score": 0}
            st.session_state.algorithm_stats[current_algorithm]["count"] += 1
            st.session_state.algorithm_stats[current_algorithm]["total_score"] += int(rating)
            
            st.success("Thanks for your rating!")

st.markdown("\n\n")
st.caption("Tip: adjust the threshold if responses feel too generic or too strict.")

# Show algorithm comparison stats
if st.session_state.ratings:
    st.markdown("---")
    st.subheader("📊 Algorithm Performance Comparison")
    
    # Overall stats
    scores = [r["score"] for r in st.session_state.ratings]
    avg = sum(scores) / len(scores)
    st.markdown(f"**Overall Average Rating:** {avg:.2f} (n={len(scores)})")
    
    # Algorithm-specific stats
    if st.session_state.algorithm_stats:
        st.markdown("**Algorithm Performance:**")
        for algo, stats in st.session_state.algorithm_stats.items():
            if stats["count"] > 0:
                avg_score = stats["total_score"] / stats["count"]
                st.markdown(f"• **{algo}**: {avg_score:.2f} (n={stats['count']})")
    
    # Show recent ratings
    if len(st.session_state.ratings) > 0:
        with st.expander("View Recent Ratings"):
            for i, rating in enumerate(st.session_state.ratings[-5:]):  # Show last 5
                st.write(f"**Turn {rating['turn']}** - {rating['algorithm']} - Score: {rating['score']}/5")
                if rating['feedback']:
                    st.caption(f"Feedback: {rating['feedback']}")
                st.markdown("---")



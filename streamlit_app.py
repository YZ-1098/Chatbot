import os
import pickle
from typing import List, Tuple, Optional
import random

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from tensorflow.keras.models import load_model


@st.cache_resource(show_spinner=False)
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    
    # Load SentenceTransformer artifacts
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
    
    # Load TensorFlow artifacts if available
    tf_model = None
    tf_vectorizer = None
    tf_label_encoder = None
    tf_responses = None
    
    tf_model_path = os.path.join(base_dir, "tensorflow_model")
    if os.path.exists(tf_model_path):
        tf_model = load_model(tf_model_path)
        with open(os.path.join(base_dir, "tf_vectorizer.pkl"), "rb") as f:
            tf_vectorizer = pickle.load(f)
        with open(os.path.join(base_dir, "tf_label_encoder.pkl"), "rb") as f:
            tf_label_encoder = pickle.load(f)
        with open(os.path.join(base_dir, "tf_responses.pkl"), "rb") as f:
            tf_responses = pickle.load(f)
    
    return model, question_embeddings, answers, questions, categories, tf_model, tf_vectorizer, tf_label_encoder, tf_responses


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

def tensorflow_predict(user_text: str, tf_model, tf_vectorizer, tf_label_encoder, tf_responses) -> Tuple[str, float]:
    """Predict using TensorFlow model"""
    if tf_model is None or tf_vectorizer is None or tf_label_encoder is None or tf_responses is None:
        return "TensorFlow model not available", 0.0
    
    # Vectorize input
    bow = tf_vectorizer.transform([user_text]).toarray()
    
    # Predict
    pred = tf_model.predict(bow, verbose=0)[0]
    confidence = float(np.max(pred))
    
    # Get intent
    intent = tf_label_encoder.inverse_transform([np.argmax(pred)])[0]
    
    # Get random response for this intent
    response = random.choice(tf_responses[intent])
    
    return response, confidence


st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

st.title("💬 University FAQ Chatbot")
st.caption("Multi-Algorithm Chatbot for Group Assignment - Compare different NLP approaches!")


model, question_embeddings, answers, questions, categories, tf_model, tf_vectorizer, tf_label_encoder, tf_responses = load_artifacts()

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

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Type your message…")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Set current algorithm for tracking
    if active_algorithm == "Yong Zheng" or active_algorithm is None:
        current_algorithm = "Yong Zheng (Deep Learning)"
    else:
        current_algorithm = active_algorithm

    # Group member algorithm-specific processing
    if active_algorithm is None:  # "(All Members)" selected
        # Show results from all algorithms
        all_results = {}
        
        # Yong Zheng's results
        yong_candidates = retrieve_top_k(prompt, model, question_embeddings, answers, top_k=top_k)
        if yong_candidates:
            all_results["Yong Zheng (Deep Learning)"] = yong_candidates[0]  # Get best result
        
        # Ew Chiu Linn's TensorFlow results
        if tf_model is not None:
            tf_response, tf_confidence = tensorflow_predict(prompt, tf_model, tf_vectorizer, tf_label_encoder, tf_responses)
            all_results["Ew Chiu Linn (TensorFlow)"] = (tf_response, tf_confidence, -1)
        else:
            all_results["Ew Chiu Linn (TensorFlow)"] = ("TensorFlow model not trained yet", 0.0, -1)
        
        # Chong Yee Yang's results (placeholder)
        all_results["Chong Yee Yang"] = ("Algorithm not implemented yet", 0.0, -1)
        
        # Display all results
        with st.chat_message("assistant"):
            st.markdown("**Responses from all group members:**")
            for member, (answer, score, idx) in all_results.items():
                if score > 0:
                    st.markdown(f"**{member}:** {answer}")
                    st.caption(f"Score: {score:.3f}")
                else:
                    st.markdown(f"**{member}:** {answer}")
                st.markdown("---")
            
            # Show candidates if enabled
            if show_candidates and yong_candidates:
                with st.expander("View all candidates from implemented algorithms"):
                    st.markdown("**Yong Zheng (Deep Learning) candidates:**")
                    for ans, score, idx in yong_candidates:
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
        
        # Use the best overall result for rating purposes
        candidates = yong_candidates if yong_candidates else []
        
    elif active_algorithm == "Yong Zheng":
        # Yong Zheng's SentenceTransformer implementation
        candidates = retrieve_top_k(prompt, model, question_embeddings, answers, top_k=top_k)
    
    elif active_algorithm == "Ew Chiu Linn":
        # Ew Chiu Linn's TensorFlow implementation
        if tf_model is not None:
            tf_response, tf_confidence = tensorflow_predict(prompt, tf_model, tf_vectorizer, tf_label_encoder, tf_responses)
            candidates = [(tf_response, tf_confidence, -1)]
        else:
            st.warning("🚧 TensorFlow model not trained yet. Run 'python train_tensorflow.py' first.")
            candidates = []
    
    elif active_algorithm == "Chong Yee Yang":
        # Placeholder for Chong Yee Yang's implementation
        st.warning("🚧 Chong Yee Yang's algorithm implementation not yet available. Please implement this algorithm.")
        candidates = []
    
    else:
        # Fallback to Yong Zheng's implementation
        candidates = retrieve_top_k(prompt, model, question_embeddings, answers, top_k=top_k)
    
    # Handle response display based on selection
    if active_algorithm is None:  # "(All Members)" selected
        # Response already displayed above, just add to history
        combined_reply = "Responses from all group members displayed above"
        st.session_state.history.append(("assistant", combined_reply))
    else:
        # Single algorithm response
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

st.markdown("\n\n")
st.caption("Tip: adjust the threshold if responses feel too generic or too strict.")



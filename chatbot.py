import random
import pickle
import numpy as np
import time
import datetime
import os
from sentence_transformers import SentenceTransformer, util

# Load retrieval artifacts (resolve relative to this file's directory)
BASE_DIR = os.path.dirname(__file__)
model = SentenceTransformer(os.path.join(BASE_DIR, "sbert_model"))
with open(os.path.join(BASE_DIR, "qa_embeddings.pkl"), "rb") as f:
    question_embeddings = pickle.load(f)
with open(os.path.join(BASE_DIR, "qa_answers.pkl"), "rb") as f:
    answers = pickle.load(f)


def retrieve_answer(user_text: str, top_k: int = 1, threshold: float = 0.35):
    user_embedding = model.encode([user_text], convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings).flatten()
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    if best_score < threshold:
        return None, best_score
    return answers[best_idx], best_score

# -------------------------
# Chat loop (retrieval)
# -------------------------

print("Chatbot is running! (type 'quit' to exit)\n")

hour = datetime.datetime.now().hour
if hour < 12:
    print("Chatbot: Good morning! How can I assist you today?")
elif hour < 18:
    print("Chatbot: Good afternoon! How can I help?")
else:
    print("Chatbot: Good evening! How may I help you today?")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Chatbot: Goodbye! Have a great day! 👋")
        break

    reply, score = retrieve_answer(message)
    time.sleep(0.8)
    if reply is None:
        print("Chatbot: I’m not sure yet. Could you rephrase or ask something else?")
    else:
        print(f"Chatbot: {reply}")

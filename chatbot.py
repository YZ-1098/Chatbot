import random
import pickle
import numpy as np
import time
import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Load retrieval artifacts
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
qa_matrix = pickle.load(open("qa_matrix.pkl", "rb"))
answers = pickle.load(open("qa_answers.pkl", "rb"))


def retrieve_answer(user_text: str, top_k: int = 1, threshold: float = 0.35):
    user_vec = vectorizer.transform([user_text])
    sims = cosine_similarity(user_vec, qa_matrix).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
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

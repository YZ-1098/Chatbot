import json
import pickle
import os
from typing import List, Tuple, Optional

from sentence_transformers import SentenceTransformer, util

# -------------------------
# Load intents.json
# -------------------------
def read_intents_json(json_path: str) -> Tuple[List[str], List[str], Optional[List[str]]]:
    questions, answers, categories = [], [], []
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    for item in data.get('intents', []):
        intent_name = item.get('intent', '').strip()
        texts = item.get('text', [])
        responses = item.get('responses', [])
        if not texts or not responses:
            continue
        canonical_answer = str(responses[0]).strip()
        for t in texts:
            q = str(t).strip()
            if not q:
                continue
            questions.append(q)
            answers.append(canonical_answer)
            categories.append(intent_name)
    return questions, answers, (categories if any(categories) else None)


# -------------------------
# Train embeddings
# -------------------------
def train_embeddings(questions: List[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight BERT variant
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    return model, question_embeddings


# -------------------------
# Save artifacts
# -------------------------
def save_artifacts(model, embeddings, answers, questions, categories, out_dir="."):
    model.save(os.path.join(out_dir, "sbert_model"))
    with open(os.path.join(out_dir, "qa_embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)
    with open(os.path.join(out_dir, "qa_answers.pkl"), "wb") as f:
        pickle.dump(answers, f)
    with open(os.path.join(out_dir, "qa_questions.pkl"), "wb") as f:
        pickle.dump(questions, f)
    if categories is not None:
        with open(os.path.join(out_dir, "qa_categories.pkl"), "wb") as f:
            pickle.dump(categories, f)


if __name__ == "__main__":
    if not os.path.exists("intents.json"):
        raise FileNotFoundError("intents.json not found")

    questions, answers, categories = read_intents_json("intents.json")
    model, embeddings = train_embeddings(questions)
    save_artifacts(model, embeddings, answers, questions, categories)
    print(f"Trained deep learning retriever on {len(questions)} Q/A pairs and saved artifacts.")

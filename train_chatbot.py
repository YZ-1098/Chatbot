import json
import pickle
import os
from typing import List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_intents_json(json_path: str) -> Tuple[List[str], List[str], Optional[List[str]]]:
    questions: List[str] = []
    answers: List[str] = []
    categories: List[str] = []
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    intents = data.get('intents') or []
    for item in intents:
        intent_name = (item.get('intent') or '').strip()
        texts = item.get('text') or []
        responses = item.get('responses') or []
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
    if not questions:
        raise ValueError("No usable entries found in intents.json")
    return questions, answers, (categories if any(categories) else None)


def train_retriever(questions: List[str]):
    # Character n-grams are robust to spelling/wording variations
    vectorizer = TfidfVectorizer(lowercase=True,
                                 analyzer='char_wb',
                                 ngram_range=(3, 5),
                                 min_df=1)
    question_matrix = vectorizer.fit_transform(questions)
    return vectorizer, question_matrix


def save_artifacts(vectorizer, question_matrix, answers: List[str], questions: List[str], categories: Optional[List[str]], out_dir: str = "."):
    with open(os.path.join(out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(out_dir, "qa_answers.pkl"), "wb") as f:
        pickle.dump(answers, f)
    # Pickle sparse matrix (requires scipy in sklearn deps, safe to pickle)
    with open(os.path.join(out_dir, "qa_matrix.pkl"), "wb") as f:
        pickle.dump(question_matrix, f)
    # Also save questions for display and optional categories for filtering
    with open(os.path.join(out_dir, "qa_questions.pkl"), "wb") as f:
        pickle.dump(questions, f)
    if categories is not None:
        with open(os.path.join(out_dir, "qa_categories.pkl"), "wb") as f:
            pickle.dump(categories, f)


if __name__ == "__main__":

    if os.path.exists("intents.json"):
        questions, answers, categories = read_intents_json("intents.json")
        source = "intents.json"
    else:
        csv_path = "Conversation.csv"
        if not os.path.exists(csv_path):
            # Allow alternate provided filename just in case
            alt = "Conversations.csv"
            if os.path.exists(alt):
                csv_path = alt
            else:
                raise FileNotFoundError("Neither intents.json nor Conversation.csv found in project root")
        questions, answers, categories = read_conversation_csv(csv_path)
        source = csv_path

    vectorizer, question_matrix = train_retriever(questions)
    save_artifacts(vectorizer, question_matrix, answers, questions, categories)
    extra = " with categories" if categories is not None else ""
    print(f"Loaded from {source}. Trained retrieval model on {len(questions)} Q/A pairs{extra} and saved artifacts.")

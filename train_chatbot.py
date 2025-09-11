import csv
import pickle
import os
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_conversation_csv(csv_path: str) -> Tuple[List[str], List[str]]:
    questions: List[str] = []
    answers: List[str] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Expecting columns: question, answer
        for row in reader:
            q = (row.get('question') or '').strip()
            a = (row.get('answer') or '').strip()
            if not q or not a:
                continue
            questions.append(q)
            answers.append(a)
    if not questions:
        raise ValueError("No question/answer rows found in Conversation.csv")
    return questions, answers


def train_retriever(questions: List[str]):
    # Character n-grams are robust to spelling/wording variations
    vectorizer = TfidfVectorizer(lowercase=True,
                                 analyzer='char_wb',
                                 ngram_range=(3, 5),
                                 min_df=1)
    question_matrix = vectorizer.fit_transform(questions)
    return vectorizer, question_matrix


def save_artifacts(vectorizer, question_matrix, answers: List[str], out_dir: str = "."):
    with open(os.path.join(out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(out_dir, "qa_answers.pkl"), "wb") as f:
        pickle.dump(answers, f)
    # Pickle sparse matrix (requires scipy in sklearn deps, safe to pickle)
    with open(os.path.join(out_dir, "qa_matrix.pkl"), "wb") as f:
        pickle.dump(question_matrix, f)


if __name__ == "__main__":

    csv_path = "Conversation.csv"
    if not os.path.exists(csv_path):
        # Allow alternate provided filename just in case
        alt = "Conversations.csv"
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError("Conversation.csv not found in project root")

    questions, answers = read_conversation_csv(csv_path)
    vectorizer, question_matrix = train_retriever(questions)
    save_artifacts(vectorizer, question_matrix, answers)
    print(f"Trained retrieval model on {len(questions)} Q/A pairs and saved artifacts.")

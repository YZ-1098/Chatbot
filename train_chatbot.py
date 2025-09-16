import json
import pickle
import os
from typing import List, Tuple, Optional
import numpy as np

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
except Exception:
    corpus_bleu = None
    SmoothingFunction = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

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


# -------------------------
# Evaluation functions (from evaluate_chatbot.py)
# -------------------------
def evaluate_classification(questions: List[str], gold_answers: List[str], model, question_embeddings, index_to_answer: List[str], threshold: float = 0.35):
    y_true: List[int] = []
    y_pred: List[int] = []

    answer_to_index = {}
    for idx, ans in enumerate(index_to_answer):
        answer_to_index.setdefault(ans, idx)

    for q, gold in zip(questions, gold_answers):
        user_embedding = model.encode([q], convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, question_embeddings).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        pred_idx = best_idx if best_score >= threshold else -1

        true_idx = answer_to_index.get(gold, -1)
        y_true.append(true_idx)
        y_pred.append(pred_idx)

    y_true_bin = [1 if t != -1 else 0 for t in y_true]
    y_pred_bin = [1 if (p != -1 and index_to_answer[p] == gold_answers[i]) else 0 for i, p in enumerate(y_pred)]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='binary', zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(sum(y_true_bin)),
    }


def evaluate_generation(questions: List[str], gold_answers: List[str], model, question_embeddings, index_to_answer: List[str]):
    preds: List[str] = []
    for q in questions:
        user_embedding = model.encode([q], convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, question_embeddings).flatten()
        best_idx = int(np.argmax(similarities))
        preds.append(index_to_answer[best_idx])

    results = {}

    if corpus_bleu is not None:
        smoothie = SmoothingFunction().method3 if SmoothingFunction else None
        references = [[g.split()] for g in gold_answers]
        hypotheses = [p.split() for p in preds]
        try:
            bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
            results["bleu"] = float(bleu)
        except Exception:
            results["bleu"] = None
    else:
        results["bleu"] = None

    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_f = []
        r2_f = []
        rl_f = []
        for ref, hyp in zip(gold_answers, preds):
            scores = scorer.score(ref, hyp)
            r1_f.append(scores["rouge1"].fmeasure)
            r2_f.append(scores["rouge2"].fmeasure)
            rl_f.append(scores["rougeL"].fmeasure)
        results["rouge1_f"] = float(np.mean(r1_f)) if r1_f else None
        results["rouge2_f"] = float(np.mean(r2_f)) if r2_f else None
        results["rougeL_f"] = float(np.mean(rl_f)) if rl_f else None
    else:
        results["rouge1_f"] = None
        results["rouge2_f"] = None
        results["rougeL_f"] = None

    return results


if __name__ == "__main__":
    if not os.path.exists("intents.json"):
        raise FileNotFoundError("intents.json not found")

    # Training phase
    questions, answers, categories = read_intents_json("intents.json")
    model, embeddings = train_embeddings(questions)
    save_artifacts(model, embeddings, answers, questions, categories)
    print(f"Trained deep learning retriever on {len(questions)} Q/A pairs and saved artifacts.")
    
    # Evaluation phase
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    cls_metrics = evaluate_classification(questions, answers, model, embeddings, answers)
    gen_metrics = evaluate_generation(questions, answers, model, embeddings, answers)
    
    print("Classification metrics (exact-match on answer text):")
    print(f"  Precision: {cls_metrics['precision']:.3f}")
    print(f"  Recall:    {cls_metrics['recall']:.3f}")
    print(f"  F1-score:  {cls_metrics['f1']:.3f}")
    print(f"  Support:   {cls_metrics['support']}")
    
    print("\nGeneration metrics:")
    if gen_metrics['bleu'] is not None:
        print(f"  BLEU:      {gen_metrics['bleu']:.3f}")
    else:
        print("  BLEU:      N/A (nltk not available)")
    
    if gen_metrics['rouge1_f'] is not None:
        print(f"  ROUGE-1:   {gen_metrics['rouge1_f']:.3f}")
        print(f"  ROUGE-2:   {gen_metrics['rouge2_f']:.3f}")
        print(f"  ROUGE-L:   {gen_metrics['rougeL_f']:.3f}")
    else:
        print("  ROUGE:     N/A (rouge-score not available)")
    
    print("\nTraining and evaluation completed!")

import os
import json
import pickle
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
except Exception:
    corpus_bleu = None  # type: ignore
    SmoothingFunction = None  # type: ignore

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None  # type: ignore


# CSV path removed from the project; evaluation now uses intents.json only


def read_intents_json(json_path: str) -> Tuple[List[str], List[str]]:
    questions: List[str] = []
    answers: List[str] = []
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    intents = data.get('intents') or []
    for item in intents:
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
    if not questions:
        raise ValueError("No usable entries found in intents.json")
    return questions, answers


def load_artifacts(base_dir: str):
    model = SentenceTransformer(os.path.join(base_dir, "sbert_model"))
    with open(os.path.join(base_dir, "qa_embeddings.pkl"), "rb") as f:
        question_embeddings = pickle.load(f)
    with open(os.path.join(base_dir, "qa_answers.pkl"), "rb") as f:
        answers = pickle.load(f)
    return model, question_embeddings, answers


def evaluate_classification(
    questions: List[str],
    gold_answers: List[str],
    model,
    question_embeddings,
    index_to_answer: List[str],
    threshold: float = 0.35,
):
    y_true: List[int] = []
    y_pred: List[int] = []

    # Gold answer to index mapping (first occurrence wins)
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

    # Convert to binary correctness for strict exact-match on answer text
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


def evaluate_generation(
    questions: List[str],
    gold_answers: List[str],
    model,
    question_embeddings,
    index_to_answer: List[str],
):
    # Build predictions (top-1)
    preds: List[str] = []
    for q in questions:
        user_embedding = model.encode([q], convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, question_embeddings).flatten()
        best_idx = int(np.argmax(similarities))
        preds.append(index_to_answer[best_idx])

    results = {}

    # BLEU (requires nltk)
    if corpus_bleu is not None:
        smoothie = SmoothingFunction().method3 if SmoothingFunction else None
        # corpus_bleu expects references as list of list of tokens; hypotheses as list of tokens
        references = [[g.split()] for g in gold_answers]
        hypotheses = [p.split() for p in preds]
        try:
            bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
            results["bleu"] = float(bleu)
        except Exception:
            results["bleu"] = None
    else:
        results["bleu"] = None

    # ROUGE (requires rouge-score)
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


def main():
    base_dir = os.path.dirname(__file__)
    questions, gold_answers = read_intents_json(os.path.join(base_dir, "intents.json"))
    print(f"Evaluating on {len(questions)} items from intents.json")

    model, question_embeddings, answers = load_artifacts(base_dir)

    cls_metrics = evaluate_classification(questions, gold_answers, model, question_embeddings, answers)
    gen_metrics = evaluate_generation(questions, gold_answers, model, question_embeddings, answers)

    print("Classification (exact-match on answer text):")
    print(cls_metrics)
    print("\nGeneration metrics:")
    print(gen_metrics)


if __name__ == "__main__":
    main()



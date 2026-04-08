import json
import argparse
import string
from collections import Counter

import regex as re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def normalize(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    if s is None:
        return None

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ''.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em(answers1, answers2):
    """Exact match after normalization."""
    if isinstance(answers1, str):
        answers1 = [answers1]
    if isinstance(answers2, str):
        answers2 = [answers2]

    if answers1 is None or answers2 is None:
        return False

    answers1 = set(normalize(a) for a in answers1 if a)
    answers2 = set(normalize(a) for a in answers2 if a)

    for a1 in answers1:
        for a2 in answers2:
            if a1 == a2:
                return True
    return False

def recall(pred_answers, gold_answers):
    pred_answers = {normalize(a) for a in pred_answers if a is not None}
    gold_answers = {normalize(a) for a in gold_answers if a is not None}
    return len(pred_answers.intersection(gold_answers))

def precision(pred_answers, gold_answers):
    pred_answers = {normalize(a) for a in pred_answers if a is not None}
    gold_answers = {normalize(a) for a in gold_answers if a is not None}
    return len(pred_answers.intersection(gold_answers))

def precision_recall(pred_answers, gold_answers):
    micro_rec = recall(pred_answers, gold_answers)
    micro_pre = precision(pred_answers, gold_answers)
    macro_rec = micro_rec / len(gold_answers) if len(gold_answers) > 0 else 0
    macro_pre = micro_pre / len(pred_answers) if len(pred_answers) > 0 else 0

    return {
        "macro_f1": 2 / (1 / macro_rec + 1 / macro_pre) if macro_rec and macro_pre else 0,
        "macro_rec": macro_rec,
        "micro_rec": micro_rec,
        "micro_pre": micro_pre,
        "micro_pred_total": len(pred_answers),
        "micro_gold_total": len(gold_answers),
    }

def eval_respond(data, answers_key):
    greedy_metrics_counter = Counter()
    sample_metrics_counter = Counter()

    for ex in data:
        gold_answers = set(ex[answers_key])

        greedy_preds = [ex["pred"]["response"]]
        sample_preds = [
            r for r, _ in Counter(
                [normalize(s) for s in ex["pred"].get("response_samples", [])]
            ).most_common(len(gold_answers))
        ]
    greedy_metrics_counter.update(precision_recall(greedy_preds, gold_answers))
        sample_metrics_counter.update(precision_recall(sample_preds, gold_answers))
        greedy_metrics_counter["any_em"] += em(greedy_preds, gold_answers)
        sample_metrics_counter["any_em"] += em(sample_preds, gold_answers)

    greedy_metrics = {
        "rec": greedy_metrics_counter["macro_rec"] / len(data),
        "f1": greedy_metrics_counter["macro_f1"] / len(data),
        "em": greedy_metrics_counter["any_em"] / len(data),
    }
    sample_metrics = {
        "rec": sample_metrics_counter["micro_rec"] / len(data),
        "f1": sample_metrics_counter["macro_f1"] / len(data),
        "em": sample_metrics_counter["any_em"] / len(data),
    }

    print("Greedy Metrics:")
    keys, vals = zip(*greedy_metrics.items())
    print("\t".join(keys))
    print("\t".join([f"{v:.3f}" for v in vals]))

    print("Sample Metrics:")
    keys, vals = zip(*sample_metrics.items())
    print("\t".join(keys))
    print("\t".join([f"{v:.3f}" for v in vals]))

    return {"greedy": greedy_metrics, "sample": sample_metrics}

def eval_clarify_q(data):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    bleu_scores = []
    rouge_scores = []

    for ex in data:
        gold = ex["clarification"]["question"]
        pred = ex["pred"]["clarification"]["question"]

        if pred is None:
            pred = ""

        ref_tokens = gold.lower().split()
        pred_tokens = pred.lower().split()

        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)

        rouge = scorer.score(gold, pred)
        rouge_scores.append(rouge["rougeL"].fmeasure)

    metrics = {
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
    }

    keys, vals = zip(*metrics.items())
    print("\t".join(keys))
    print("\t".join([f"{v:.3f}" for v in vals]))
    return metrics

def eval_clarify(data, answers_key):
    metrics_counter = Counter()

    for ex in data:
        em_count = 0
        gold_answer_count = 0

        for answer in ex["pred"]["clarification"]["eval_answers"]:
            if not answer[answers_key]:
                continue

            em_count += bool(
                em(answer["response"], answer["gold_response"]) and
                answer["answer"] and
                (
                    normalize(answer["gold_response"]) not in normalize(answer["answer"])
                    or normalize(answer["gold_response"]) not in normalize(ex["pred"]["clarification"]["question"])
                )
            )
            gold_answer_count += 1

        if gold_answer_count > 0:
            metrics_counter["macro_f1"] += em_count / gold_answer_count
        metrics_counter["micro_f1"] += em_count
        metrics_counter["micro_total"] += gold_answer_count
        metrics_counter["any_em"] += bool(em_count)

    metrics = {
        "f1": metrics_counter["macro_f1"] / len(data) if len(data) > 0 else 0,
        "em": metrics_counter["any_em"] / len(data) if len(data) > 0 else 0,
    }

    keys, vals = zip(*metrics.items())
    print("\t".join(keys))
    print("\t".join([f"{v:.3f}" for v in vals]))

    return metrics

def main(args):
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    ambig_data = [ex for ex in data if ex.get("isambig")]

    results = {}

    if args.mode == "respond":
        print("NQ-Open Evaluations:")
        results["nq_open"] = eval_respond(data, "nq_answers")
        print()
        print("AmbigQA Evaluations:")
        results["ambigqa"] = eval_respond(ambig_data, "answers")

    elif args.mode == "clarify_q":
        print("Clarifying Question Quality (all):")
        results["all"] = eval_clarify_q(data)
        print()
        print("Clarifying Question Quality (ambiguous only):")
        results["ambig"] = eval_clarify_q(ambig_data)

    elif args.mode == "clarify":
        print("NQ-Open Evaluations:")
        results["nq_open"] = eval_clarify(data, "is_nq")
        print()
        print("AmbigQA Evaluations:")
        results["ambigqa"] = eval_clarify(ambig_data, "is_ambig")

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print()

    metrics_path = args.input_path.replace(".jsonl", ".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--mode", required=True, choices=["respond", "clarify_q", "clarify"])
    args = parser.parse_args()
    main(args)

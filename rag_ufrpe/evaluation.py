"""Mede recuperação e abstenção. A qualidade da geração exige revisão separada."""

import csv
import json
from datetime import date

from .corpus import ROOT
from .retrieval import Retriever

K = 3
NOTE = ("Amostra autoral pequena. As métricas não avaliam a correção da resposta gerada. "
        "Rótulos sujeitos a revisão independente.")


def ids_of(value):
    return set(filter(None, value.split(";")))


def rank_metrics(retrieved, relevant, k=K):
    """Métricas de uma pergunta respondível. `retrieved` é a lista ordenada de ids."""
    retrieved = retrieved[:k]
    overlap = len(set(retrieved) & relevant)
    precision = overlap / k
    recall = overlap / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    first_hit = next((rank for rank, item in enumerate(retrieved, start=1) if item in relevant), None)
    return {
        "precision_at_3": precision,
        # A precisão@3 de quem retorna só um trecho não passa de 1/3. Esta divide pelo
        # número de trechos de fato retornados.
        "precision_returned": overlap / len(retrieved) if retrieved else 0.0,
        "recall_at_3": recall, "f1_at_3": f1,
        "hit_at_3": bool(overlap), "hit_at_1": bool(retrieved and retrieved[0] in relevant),
        "reciprocal_rank": 1 / first_hit if first_hit else 0.0,
    }


def load_cases(split):
    with (ROOT / "avaliacao" / "perguntas.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    splits = {row["split"] for row in rows}
    if split != "all" and split not in splits:
        raise ValueError(f"Split deve ser all ou um de: {', '.join(sorted(splits))}.")
    return [row for row in rows if split == "all" or row["split"] == split]


def evaluate_retrieval(split="test", retriever=None, write=True):
    cases = load_cases(split)
    retriever = retriever or Retriever()
    known_ids = {chunk["id"] for chunk in retriever.chunks}
    for case in cases:
        if not ids_of(case["expected_ids"]) <= known_ids:
            raise ValueError(f"Rótulo aponta para trecho inexistente no caso {case['id']}.")
    results = []
    for case in cases:
        found = retriever.search(case["question"], case["semester"] or None, k=K)
        retrieved = [item["id"] for item in found["sources"]]
        relevant = ids_of(case["expected_ids"])
        results.append({
            "id": case["id"], "question": case["question"], "split": case["split"],
            "answerable": bool(relevant), "expected_status": case["expected_status"],
            "status": found["status"], "status_correct": found["status"] == case["expected_status"],
            "expected_ids": case["expected_ids"], "retrieved_ids": ";".join(retrieved),
            **rank_metrics(retrieved, relevant),
        })
    answerable = [item for item in results if item["answerable"]]
    abstentions = [item for item in results if not item["answerable"]]

    def mean(rows, key):
        return round(sum(item[key] for item in rows) / len(rows), 4) if rows else None

    summary = {
        "date": str(date.today()), "split": split, "k": K, "cases": len(results),
        "answerable_cases": len(answerable), "abstention_cases": len(abstentions),
        "macro_precision_at_3": mean(answerable, "precision_at_3"),
        "macro_precision_returned": mean(answerable, "precision_returned"),
        "macro_recall_at_3": mean(answerable, "recall_at_3"),
        "macro_f1_at_3": mean(answerable, "f1_at_3"),
        "hit_at_3": mean(answerable, "hit_at_3"),
        "hit_at_1": mean(answerable, "hit_at_1"),
        "mrr_at_3": mean(answerable, "reciprocal_rank"),
        "answerable_status_accuracy": mean(answerable, "status_correct"),
        "abstention_status_accuracy": mean(abstentions, "status_correct"),
        "note": NOTE,
    }
    if write:
        target = ROOT / "avaliacao" / f"resultados_{split}.csv"
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        (ROOT / "avaliacao" / f"metricas_{split}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def generate_examples(split="all", backend="local", retriever=None):
    """Registra as respostas completas para revisão humana. Não mede correção."""
    from .generation import answer
    from .retrieval import retrieve

    retriever = retriever or Retriever()
    examples = []
    for case in load_cases(split):
        if case["expected_status"] not in {"ok", "no_evidence"}:
            continue
        result = answer(retrieve(case["question"], case["semester"] or None, lambda: retriever),
                        case["question"], backend)
        examples.append({
            "id": case["id"], "question": case["question"], "expected_answer": case["expected_answer"],
            "answer": result["answer"], "mode": result["mode"],
            "source_ids": [item["id"] for item in result["sources"]],
            "review_status": "pendente_revisao_independente",
        })
    target = ROOT / "avaliacao" / "geracao_exemplos.json"
    target.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    return examples

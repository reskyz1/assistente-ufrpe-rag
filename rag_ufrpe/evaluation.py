"""Mede recuperação e abstenção. A qualidade da geração exige revisão separada."""

import csv
import json
from datetime import date

from .corpus import ROOT
from .retrieval import Retriever


def evaluate_retrieval(split="test", retriever=None):
    if split not in {"dev", "test", "all"}:
        raise ValueError("Split deve ser dev, test ou all.")
    with (ROOT / "avaliacao" / "perguntas.csv").open(encoding="utf-8", newline="") as stream:
        cases = [row for row in csv.DictReader(stream) if split == "all" or row["split"] == split]
    retriever = retriever or Retriever()
    known_ids = {chunk["id"] for chunk in retriever.chunks}
    for case in cases:
        relevant = set(filter(None, case["expected_ids"].split(";")))
        if not relevant <= known_ids:
            raise ValueError(f"Rótulo aponta para trecho inexistente no caso {case['id']}.")
    results = []
    for case in cases:
        found = retriever.search(case["question"], case["semester"] or None, k=3)
        predicted = {item["id"] for item in found["sources"]}
        relevant = set(filter(None, case["expected_ids"].split(";")))
        overlap = len(predicted & relevant)
        precision = overlap / 3
        recall = overlap / len(relevant) if relevant else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append({"id": case["id"], "question": case["question"], "split": case["split"],
                        "answerable": bool(relevant), "expected_status": case["expected_status"],
                        "status": found["status"], "status_correct": found["status"] == case["expected_status"],
                        "expected_ids": case["expected_ids"], "retrieved_ids": ";".join(item["id"] for item in found["sources"]),
                        "precision_at_3": precision, "recall_at_3": recall, "f1_at_3": f1,
                        "hit_at_3": bool(overlap),
                        "hit_at_1": bool(found["sources"] and found["sources"][0]["id"] in relevant)})
    answerable = [item for item in results if item["answerable"]]
    abstentions = [item for item in results if not item["answerable"]]
    def mean(rows, key):
        return round(sum(item[key] for item in rows) / len(rows), 4) if rows else None
    summary = {
        "date": str(date.today()), "split": split, "k": 3, "cases": len(results),
        "answerable_cases": len(answerable), "abstention_cases": len(abstentions),
        "macro_precision_at_3": mean(answerable, "precision_at_3"),
        "macro_recall_at_3": mean(answerable, "recall_at_3"),
        "macro_f1_at_3": mean(answerable, "f1_at_3"),
        "hit_at_3": mean(answerable, "hit_at_3"),
        "hit_at_1": mean(answerable, "hit_at_1"),
        "abstention_status_accuracy": mean(abstentions, "status_correct"),
        "note": "Amostra autoral pequena. As métricas não avaliam a correção da resposta gerada. Rótulos sujeitos a revisão independente.",
    }
    target = ROOT / "avaliacao" / f"resultados_{split}.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (ROOT / "avaliacao" / f"metricas_{split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

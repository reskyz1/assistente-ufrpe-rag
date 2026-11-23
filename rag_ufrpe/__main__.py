"""Comandos para preparar os dados, perguntar e avaliar a recuperação."""

import argparse
import json
import os

from dotenv import load_dotenv

from .corpus import ROOT, build_corpus


def main():
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    parser = argparse.ArgumentParser(description="Assistente RAG experimental da UFRPE")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preparar", help="Confere os PDFs e extrai os trechos")
    query = commands.add_parser("perguntar", help="Consulta os documentos")
    query.add_argument("question")
    query.add_argument("--semestre", choices=["2026.1", "2026.2"])
    query.add_argument("--backend", choices=["local", "ollama", "sources"], default="local")
    query.add_argument("--json", action="store_true")
    evaluate = commands.add_parser("avaliar", help="Avalia a busca, sem avaliar o texto gerado")
    evaluate.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = parser.parse_args()
    if args.command == "preparar":
        print(f"Base preparada com {len(build_corpus())} trechos.")
        return
    if args.command == "avaliar":
        from .evaluation import evaluate_retrieval
        print(json.dumps(evaluate_retrieval(args.split), ensure_ascii=False, indent=2))
        return
    from .generation import answer
    from .retrieval import Retriever, question_context
    context = question_context(args.question, args.semestre)
    retrieval = {**context, "sources": []}
    if context["status"] == "ok":
        retrieval = Retriever().search(args.question, args.semestre)
    result = answer(retrieval, args.question, args.backend)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["answer"])
    for number, source in enumerate(result["sources"], start=1):
        print(f"\n[{number}] {source['title']} - páginas {source['pages']}\n{source['text']}\n{source['url']}")


if __name__ == "__main__":
    main()

"""Geração com modelo local ou servidor Ollama e conferência das citações."""

import json
import os
import re
from functools import lru_cache

import requests

from .corpus import normalize

SYSTEM = "Responda em português usando somente a fonte fornecida. Retorne somente JSON com as chaves resposta (texto) e fontes (lista de números)."


def make_messages(question, sources):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Fonte [1]: {sources[0]['text']}\nPergunta: {question}\n"
         'Responda no formato JSON: {"resposta": "...", "fontes": [1]}'},
    ]


@lru_cache(maxsize=1)
def load_generator(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    return tokenizer, model


def local_generate(messages):
    import torch

    model_id = os.getenv("GENERATION_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer, model = load_generator(model_id)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=280, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def ollama_generate(messages):
    response = requests.post(os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat",
                             json={"model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
                                   "messages": messages, "stream": False, "format": "json",
                                   "options": {"temperature": 0, "num_predict": 280}},
                             timeout=(10, 180))
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def time_quantities(text):
    words = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
             "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9,
             "dez": 10, "onze": 11, "doze": 12}
    pattern = r"\b(\d+|" + "|".join(words) + r")\s*(?:\([^)]*\)\s*)?(dias?|horas?|semanas?)\b"
    return {(int(value) if value.isdigit() else words[value], unit.rstrip("s"))
            for value, unit in re.findall(pattern, normalize(text))}


def compatible_numbers(text, source, question):
    stated = {int(number) for number in re.findall(r"\b\d+\b", text)}
    available = {int(number) for number in re.findall(r"\b\d+\b", source)}
    if not stated <= available:
        return False
    if re.search(r"quanto tempo|quantos dias|quantas horas|quantas semanas", normalize(question)):
        quantities = time_quantities(text)
        return bool(quantities) and quantities <= time_quantities(source)
    return True


def contradicts_prohibition(text, source):
    prohibition = re.search(r"nao pode|nao e permitid|e vedad", normalize(source))
    return bool(prohibition and re.match(r"\s*sim\b", normalize(text)))


def answer(retrieval, question, backend="local"):
    if backend not in {"local", "ollama", "sources"}:
        raise ValueError("Backend deve ser local, ollama ou sources.")
    if retrieval["status"] != "ok":
        return {**retrieval, "answer": retrieval["message"], "mode": "abstention"}
    if backend == "sources":
        return {**retrieval, "answer": "Trechos encontrados para sua conferência.", "mode": "sources"}
    try:
        # A geração usa uma única evidência, reduzindo a mistura entre regras distintas.
        # Os demais resultados continuam visíveis para conferência pelo usuário.
        messages = make_messages(question, retrieval["sources"][:1])
        raw = local_generate(messages) if backend == "local" else ollama_generate(messages)
    except (requests.RequestException, OSError, RuntimeError, ValueError, KeyError) as exc:
        # A falha permanece visível e os trechos continuam disponíveis.
        return {**retrieval, "answer": "O gerador está indisponível. Confira os trechos recuperados abaixo.",
                "mode": "generation_error", "error_type": type(exc).__name__}
    try:
        payload = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
        text = payload["resposta"]
        citations = payload["fontes"]
        if not isinstance(text, str) or not isinstance(citations, list):
            raise ValueError("Formato inválido.")
    except (ValueError, KeyError, TypeError):
        return {**retrieval, "answer": "A resposta gerada não apresentou citações válidas. Confira os trechos recuperados abaixo.",
                "mode": "invalid_citations"}
    if re.search(r"nao (?:encontrei|ha|fornece|contem).*informa", normalize(text)):
        return {**retrieval, "answer": text, "mode": "abstention"}
    if not text or not citations or any(type(number) is not int or number != 1 for number in citations):
        return {**retrieval, "answer": "A resposta gerada não apresentou citações válidas. Confira os trechos recuperados abaixo.",
                "mode": "invalid_citations"}
    source = retrieval["sources"][0]
    if not compatible_numbers(text, source["text"], question) or contradicts_prohibition(text, source["text"]):
        return {**retrieval, "answer": f"Confira a regra completa no trecho original.\n\n> {source['text']} [1]",
                "mode": "grounding_warning", "generation_source_id": source["id"]}
    references = " ".join(f"[{number}]" for number in sorted(set(citations)))
    text = f"{text} {references}"
    if source.get("kind") == "calendar":
        text += "\n\nData registrada na fonte: " + source["text"].split(". ", 1)[0] + "."
    return {**retrieval, "answer": text, "mode": "generated",
            "cited_sources": sorted(set(citations)), "generation_source_id": retrieval["sources"][0]["id"]}

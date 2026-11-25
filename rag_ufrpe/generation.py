"""Geração com modelo local ou servidor Ollama e conferência das citações."""

import json
import os
import re
from functools import lru_cache

import requests

from .corpus import normalize

ABSTENTION = "A fonte não traz informação sobre isso."
SYSTEM = ("Responda em português usando somente a fonte fornecida. Retorne somente JSON com as chaves "
          "resposta (texto) e fontes (lista de números). Se a fonte não disser claramente o que foi "
          f'perguntado, não use conhecimento próprio: responda {{"resposta": "{ABSTENTION}", "fontes": []}}.')
FORMAT = 'Responda no formato JSON: {"resposta": "...", "fontes": [1]}'
# Um exemplo de resposta e um de abstenção, com texto fictício. Só o de abstenção
# fazia o modelo pequeno recusar perguntas que a fonte responde.
EXAMPLE = [
    {"role": "user", "content": "Fonte [1]: Art. 1. A secretaria funciona de segunda a sexta, das 8h às 17h.\n"
     f"Pergunta: Qual o horário da secretaria?\n{FORMAT}"},
    {"role": "assistant", "content": json.dumps(
        {"resposta": "A secretaria funciona de segunda a sexta, das 8h às 17h.", "fontes": [1]}, ensure_ascii=False)},
    {"role": "user", "content": "Fonte [1]: Art. 1. A secretaria funciona de segunda a sexta, das 8h às 17h.\n"
     f"Pergunta: Quanto custa o estacionamento?\n{FORMAT}"},
    {"role": "assistant", "content": json.dumps({"resposta": ABSTENTION, "fontes": []}, ensure_ascii=False)},
]
# Palavras funcionais ou de moldura que não indicam conteúdo inventado.
FUNCTION_WORDS = set("""para como mais entre sobre pela pelo pelos pelas esta este essa esse isso quando onde deve pode
podem seja sera estao ainda apos cada qual quais tambem assim desde tem ter nao sao dos das que com uma uns umas por seu
sua seus suas ele ela eles elas mas ate foi sem sob voce voces conforme fonte mencionado""".split())
UNSUPPORTED_LIMIT = .45


def make_messages(question, sources):
    return [
        {"role": "system", "content": SYSTEM}, *EXAMPLE,
        {"role": "user", "content": f"Fonte [1]: {sources[0]['text']}\nPergunta: {question}\n{FORMAT}"},
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


NUMBER_WORDS = {"dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6,
                "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12}


def numbers_in(text):
    """Números em algarismos e por extenso (de dois a doze). "um" e "uma" ficam de fora por serem artigos."""
    words = re.findall(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", normalize(text))
    return {int(number) for number in re.findall(r"\b\d+\b", text)} | {NUMBER_WORDS[word] for word in words}


def time_quantities(text):
    words = {"um": 1, "uma": 1, **NUMBER_WORDS}
    pattern = r"\b(\d+|" + "|".join(words) + r")\s*(?:\([^)]*\)\s*)?(dias?|horas?|semanas?)\b"
    return {(int(value) if value.isdigit() else words[value], unit.rstrip("s"))
            for value, unit in re.findall(pattern, normalize(text))}


def compatible_numbers(text, source, question):
    if not numbers_in(text).issubset(numbers_in(source)):
        return False
    if re.search(r"quanto tempo|quantos dias|quantas horas|quantas semanas", normalize(question)):
        quantities = time_quantities(text)
        return bool(quantities) and quantities <= time_quantities(source)
    return True


def content_stems(text):
    return {word[:5] for word in re.findall(r"[a-z]{4,}", normalize(text)) if word not in FUNCTION_WORDS}


def unsupported_share(text, source, question):
    """Fração das palavras de conteúdo da resposta que não aparecem na fonte nem na pergunta.

    Detecta conteúdo inventado (por exemplo, uma lei que a fonte não cita). É uma
    heurística conservadora: não detecta afirmações falsas escritas só com palavras da fonte.
    """
    stems = content_stems(text)
    return len(stems - content_stems(source) - content_stems(question)) / len(stems) if stems else 0.0


NEGATION = re.compile(r"\bnao\b|vedad|proib|impossib|inadmis|\bsem\b|exceto|\bnunca\b")


def negation_without_support(text, source):
    """Resposta que começa com "não" quando a fonte não contém nenhuma negação.

    Pega inversões como "Não precisa de quitação" diante de uma regra que a exige.
    """
    return bool(re.match(r"\s*nao\b", normalize(text)) and not NEGATION.search(normalize(source)))


def contradicts_prohibition(text, source):
    prohibition = re.search(r"nao pode|nao e permitid|e vedad", normalize(source))
    return bool(prohibition and re.match(r"\s*sim\b", normalize(text)))


def strip_fence(raw):
    """Remove a cerca Markdown (```json ... ```) que alguns modelos acrescentam ao JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```").removeprefix("json").removesuffix("```")
    return raw.strip()


def calendar_answer(source):
    """Monta a resposta a partir do evento, no formato "descrição: Mês/Ano - data"."""
    date, _, description = source["text"].partition(". ")
    return f"{description.strip().rstrip('.')}: {date}. [1]"


def is_abstention(text, citations):
    """Abstenção explícita: a frase combinada sem fontes, ou uma negação de informação."""
    normalized = normalize(text)
    canonical = normalized.startswith(normalize(ABSTENTION).rstrip(".")) and not citations
    return canonical or bool(re.search(r"nao (?:encontrei|ha|fornece|contem|traz).*informa", normalized))


def answer(retrieval, question, backend="local"):
    if backend not in {"local", "ollama", "sources"}:
        raise ValueError("Backend deve ser local, ollama ou sources.")
    if retrieval["status"] != "ok":
        return {**retrieval, "answer": retrieval["message"], "mode": "abstention"}
    if backend == "sources":
        return {**retrieval, "answer": "Trechos encontrados para sua conferência.", "mode": "sources"}
    if retrieval["sources"][0].get("kind") == "calendar":
        # Eventos do calendário já são estruturados. Um modelo pequeno só acrescentaria risco.
        source = retrieval["sources"][0]
        return {**retrieval, "answer": calendar_answer(source), "mode": "calendar",
                "cited_sources": [1], "generation_source_id": source["id"]}
    try:
        # A geração usa uma única evidência, reduzindo a mistura entre regras distintas.
        # Os demais resultados continuam visíveis para conferência pelo usuário.
        messages = make_messages(question, retrieval["sources"][:1])
        raw = local_generate(messages) if backend == "local" else ollama_generate(messages)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:  # OSError inclui erros do requests
        # A falha permanece visível e os trechos continuam disponíveis.
        return {**retrieval, "answer": "O gerador está indisponível. Confira os trechos recuperados abaixo.",
                "mode": "generation_error", "error_type": type(exc).__name__}
    try:
        payload = json.loads(strip_fence(raw))
        text = payload["resposta"]
        citations = payload["fontes"]
        if not isinstance(text, str) or not isinstance(citations, list):
            raise ValueError("Formato inválido.")
    except (ValueError, KeyError, TypeError):
        return {**retrieval, "answer": "A resposta gerada não apresentou citações válidas. Confira os trechos recuperados abaixo.",
                "mode": "invalid_citations"}
    if is_abstention(text, citations):
        return {**retrieval, "answer": text, "mode": "abstention"}
    if not text or not citations or any(type(number) is not int or number != 1 for number in citations):
        return {**retrieval, "answer": "A resposta gerada não apresentou citações válidas. Confira os trechos recuperados abaixo.",
                "mode": "invalid_citations"}
    source = retrieval["sources"][0]
    if (not compatible_numbers(text, source["text"], question) or contradicts_prohibition(text, source["text"])
            or negation_without_support(text, source["text"])
            or unsupported_share(text, source["text"], question) > UNSUPPORTED_LIMIT):
        return {**retrieval, "answer": f"Confira a regra completa no trecho original.\n\n> {source['text']} [1]",
                "mode": "grounding_warning", "generation_source_id": source["id"]}
    references = " ".join(f"[{number}]" for number in sorted(set(citations)))
    return {**retrieval, "answer": f"{text} {references}", "mode": "generated",
            "cited_sources": sorted(set(citations)), "generation_source_id": source["id"]}

"""Busca híbrida com embeddings multilíngues e correspondência de termos."""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import ROOT, load_corpus, normalize

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
SCOPE = re.compile(r"matricul|tranc|suspens|carga horaria|ingressant|vagas|calendario|aulas|biblioteca")
DATE_QUESTION = re.compile(r"prazo|quando|\bdatas?\b|que.*dia|periodo.*(?:matricul|tranc|suspens)|ate quando")


def calendar_matches(chunk, question):
    """Distingue solicitação, processamento, rematrícula e trancamento de curso."""
    if re.search(r"suspens|tranc.*(?:curso|semestre|periodo inteiro)", question):
        return chunk["event"] == "suspensao"
    if "tranc" in question:
        return chunk["event"] == "trancamento"
    if "rematricul" in question:
        return chunk["event"] == "rematricula"
    if "extraordinaria" in question:
        return chunk["event"] == "extraordinaria"
    if "matricul" in question:
        description = normalize(chunk["text"])
        if re.search(r"processamento|ingressant|tcc|eso|orientacao|movimentacao|remt|analise", question):
            return chunk["event"] == "matricula"
        return "solicitacao de matricula dos alunos regulares" in description
    return True


def question_context(question, semester=None):
    normalized = normalize(question)
    if not question.strip():
        return {"status": "empty", "message": "Digite uma pergunta.", "semester": semester}
    if not SCOPE.search(normalized):
        return {"status": "out_of_scope", "message": "Este assistente consulta matrícula e trancamento na graduação presencial da UFRPE.", "semester": semester}
    mentioned = set(re.findall(r"20\d{2}\.[12]", question))
    if len(mentioned) > 1:
        return {"status": "clarify", "message": "Consulte um semestre de cada vez para conferir os prazos.", "semester": semester}
    explicit = next(iter(mentioned), None)
    if semester and explicit and semester != explicit:
        return {"status": "clarify", "message": "O semestre da pergunta é diferente do semestre selecionado. Ajuste a seleção.", "semester": semester}
    semester = explicit or semester
    date_question = bool(DATE_QUESTION.search(normalized))
    years = set(re.findall(r"\b20\d{2}\b", question))
    if date_question and ((semester and semester not in {"2026.1", "2026.2"}) or years - {"2026"}):
        return {"status": "unsupported_period", "message": "A base de prazos cobre somente 2026.1 e 2026.2. Consulte o calendário oficial do período solicitado.", "semester": semester}
    if date_question and not semester:
        return {"status": "clarify", "message": "Informe o semestre, 2026.1 ou 2026.2, para consultar uma data.", "semester": None}
    return {"status": "ok", "message": "", "semester": semester, "date_question": date_question}


class Retriever:
    def __init__(self, chunks=None, cache_dir=None):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks if chunks is not None else load_corpus()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.cache_dir = Path(cache_dir or ROOT / ".cache" / "index")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(json.dumps(
            {"model": EMBEDDING_MODEL, "chunks": self.chunks}, ensure_ascii=False, sort_keys=True
        ).encode()).hexdigest()
        cache = self.cache_dir / f"{fingerprint}.npy"
        passages = ["passage: " + item["text"] for item in self.chunks]
        if cache.exists():
            self.embeddings = np.load(cache, allow_pickle=False)
        else:
            self.embeddings = self.model.encode(passages, normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=False)
            # Cada processo escreve seu temporário antes de substituir o índice.
            temporary = cache.with_name(f"{fingerprint}.{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, self.embeddings, allow_pickle=False)
            temporary.replace(cache)
        self.vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)
        self.lexical = self.vectorizer.fit_transform([item["text"] for item in self.chunks])

    def search(self, question, semester=None, k=3):
        if not 1 <= k <= 10:
            raise ValueError("k deve estar entre 1 e 10.")
        context = question_context(question, semester)
        if context["status"] != "ok":
            return {**context, "sources": []}
        normalized = normalize(question)
        expanded = question
        if re.search(r"tranc.*(?:curso|semestre|periodo inteiro)", normalized):
            expanded += " suspensão de programa"
        vector = self.model.encode(["query: " + expanded], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)[0]
        semantic = self.embeddings @ vector
        lexical = (self.lexical @ self.vectorizer.transform([expanded]).T).toarray().ravel()
        # O E5 tem similaridades concentradas em valores altos. Centralizamos para
        # combinar as escalas com TF-IDF, sem interpretar o resultado como probabilidade.
        scores = .65 * np.maximum(0, (semantic - .70) / .30) + .35 * lexical
        candidates = []
        for index, chunk in enumerate(self.chunks):
            if chunk["kind"] == "calendar":
                if not context["date_question"] or chunk["semester"] != context["semester"]:
                    continue
                if not calendar_matches(chunk, normalized):
                    continue
            elif context["date_question"]:
                continue
            candidates.append(index)
        ranked = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)[:k]
        if not ranked or float(scores[ranked[0]]) < .10:
            return {**context, "status": "no_evidence", "message": "Não encontrei trechos suficientes para responder. Consulte a fonte oficial ou o setor responsável.", "sources": []}
        selected = [{**self.chunks[i], "score": round(float(scores[i]), 5),
                     "semantic_score": round(float(semantic[i]), 5)} for i in ranked]
        return {**context, "sources": selected}

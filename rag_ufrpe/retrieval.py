"""Busca híbrida com embeddings multilíngues e correspondência de termos."""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import ROOT, SUPPORTED_SEMESTERS, load_corpus, normalize

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# Pesos e deslocamento escolhidos manualmente para o protótipo. O E5 concentra as
# similaridades em valores altos (perto de 0,85), então subtraímos SEMANTIC_FLOOR.
SEMANTIC_WEIGHT, SEMANTIC_FLOOR, LEXICAL_WEIGHT = .65, .70, .35
# Este corte só descarta resultados degenerados. Os scores de perguntas fora da base
# (0,28 a 0,50) se sobrepõem aos de perguntas respondíveis (0,44 a 0,60), então ele
# NÃO detecta ausência de evidência. Ver a seção de limitações do README.
MIN_SCORE = .10

SCOPE = re.compile(
    r"matricul|tranc|suspens|carga horaria|ingressant|vagas|calendario|aulas|biblioteca|"
    r"turma|disciplina|componente curricular|discente|aluno|optativ|consolida|vinculo|"
    r"sisu|remt|ferias|semestre|periodo letivo|orientacao individual"
)
# Pergunta sobre quando algo acontece. "Data limite" também é pergunta de calendário.
DATE_QUESTION = re.compile(r"prazo|quando|\bdatas?\b|\bque dia|periodo.*(?:matricul|tranc|suspens)|ate quando")
# Pergunta sobre uma regra, mesmo contendo "prazo": "prazo máximo", "depois do prazo",
# "quando é efetivado". Tem prioridade sobre DATE_QUESTION.
RULE_HINT = re.compile(r"efetiv|maxim|limite|permitid|no total|(?:depois|apos|fora|alem) d[oa]s? (?:prazo|data|periodo)")
SEMESTER = re.compile(r"\b(20\d{2})\s*[./-]\s*([12])\b")
SEMESTER_WORDS = re.compile(r"\b([12])\s?[ºo°]?\s?(?:semestre|periodo)(?: letivo)? (?:de )?(20\d{2})\b")
# Termos que indicam um evento específico, e não a solicitação do(a) aluno(a) regular.
SPECIFIC_EVENT = re.compile(r"processamento|analise|ingressant|orientacao|movimentacao|coordena|\b(?:tcc|eso|remt)\b")


def mentioned_semesters(question):
    """Reconhece 2026.1, 2026/1, 2026-1 e "1º semestre de 2026"."""
    found = {f"{year}.{term}" for year, term in SEMESTER.findall(question)}
    found |= {f"{year}.{term}" for term, year in SEMESTER_WORDS.findall(normalize(question))}
    return found


def is_date_question(normalized):
    return bool(DATE_QUESTION.search(normalized) and not RULE_HINT.search(normalized.replace("data limite", "")))


def calendar_matches(chunk, question):
    """Distingue solicitação, processamento, rematrícula e trancamento de curso."""
    if re.search(r"suspens|tranc.*(?:curso|semestre|periodo inteiro)", question):
        return chunk["event"] == "suspensao"
    if "tranc" in question:
        return chunk["event"] == "trancamento"
    if "extraordinaria" in question:
        return chunk["event"] == "extraordinaria"
    for word, event in (("rematricul", "rematricula"), ("matricul", "matricula")):
        if word in question:
            if SPECIFIC_EVENT.search(question):
                return chunk["event"] == event
            # Sem outro detalhe, a pergunta se refere ao pedido do(a) aluno(a) regular.
            return f"solicitacao de {word}a dos alunos regulares" in normalize(chunk["text"])
    return True


def question_context(question, semester=None):
    normalized = normalize(question)
    if not question.strip():
        return {"status": "empty", "message": "Digite uma pergunta.", "semester": semester}
    if not SCOPE.search(normalized):
        return {"status": "out_of_scope", "message": "Este assistente consulta matrícula e trancamento na graduação presencial da UFRPE.", "semester": semester}
    mentioned = mentioned_semesters(question)
    if len(mentioned) > 1:
        return {"status": "clarify", "message": "Consulte um semestre de cada vez para conferir os prazos.", "semester": semester}
    explicit = next(iter(mentioned), None)
    if semester and explicit and semester != explicit:
        return {"status": "clarify", "message": "O semestre da pergunta é diferente do semestre selecionado. Ajuste a seleção.", "semester": semester}
    semester = explicit or semester
    date_question = is_date_question(normalized)
    years = set(re.findall(r"\b20\d{2}\b", question))
    supported_years = {item.split(".")[0] for item in SUPPORTED_SEMESTERS}
    if date_question and ((semester and semester not in SUPPORTED_SEMESTERS) or years - supported_years):
        return {"status": "unsupported_period", "message": f"A base de prazos cobre somente {' e '.join(SUPPORTED_SEMESTERS)}. Consulte o calendário oficial do período solicitado.", "semester": semester}
    if date_question and not semester:
        return {"status": "clarify", "message": f"Informe o semestre, {' ou '.join(SUPPORTED_SEMESTERS)}, para consultar uma data.", "semester": None}
    return {"status": "ok", "message": "", "semester": semester, "date_question": date_question}


def load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


class Retriever:
    """Índice em memória. Passe `encoder` para testar sem baixar o modelo."""

    def __init__(self, chunks=None, cache_dir=None, encoder=None):
        self.chunks = chunks if chunks is not None else load_corpus()
        self.model = encoder or load_encoder()
        passages = ["passage: " + item["text"] for item in self.chunks]
        if encoder is not None:
            # Um codificador injetado não identifica o modelo, então não usa o cache em disco.
            self.embeddings = self._encode(passages)
        else:
            self.embeddings = self._cached_embeddings(passages, Path(cache_dir or ROOT / ".cache" / "index"))
        self.vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)
        self.lexical = self.vectorizer.fit_transform([item["text"] for item in self.chunks])

    def _encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False))

    def _cached_embeddings(self, passages, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        # A impressão digital usa só o modelo, o id e o texto. Metadados como a data
        # de conferência não invalidam o índice.
        fingerprint = hashlib.sha256(json.dumps(
            {"model": EMBEDDING_MODEL, "chunks": [[item["id"], item["text"]] for item in self.chunks]},
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache = cache_dir / f"{fingerprint}.npy"
        if cache.exists():
            return np.load(cache, allow_pickle=False)
        embeddings = self._encode(passages)
        # Cada processo escreve seu temporário antes de substituir o índice.
        temporary = cache.with_name(f"{fingerprint}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, embeddings, allow_pickle=False)
        temporary.replace(cache)
        for stale in cache_dir.glob("*.npy"):
            if stale != cache:
                try:
                    stale.unlink()
                except OSError:
                    pass
        return embeddings

    @staticmethod
    def _eligible(chunk, context, normalized_question):
        """Perguntas de data só veem o calendário do semestre. As demais, só os artigos."""
        if chunk["kind"] != "calendar":
            return not context["date_question"]
        return (context["date_question"] and chunk["semester"] == context["semester"]
                and calendar_matches(chunk, normalized_question))

    def search(self, question, semester=None, k=3, context=None):
        if not 1 <= k <= 10:
            raise ValueError("k deve estar entre 1 e 10.")
        context = context or question_context(question, semester)
        if context["status"] != "ok":
            return {**context, "sources": []}
        normalized = normalize(question)
        expanded = question
        if re.search(r"tranc.*(?:curso|semestre|periodo inteiro)", normalized):
            expanded += " suspensão de programa"
        vector = self._encode(["query: " + expanded])[0]
        semantic = self.embeddings @ vector
        lexical = (self.lexical @ self.vectorizer.transform([expanded]).T).toarray().ravel()
        scores = (SEMANTIC_WEIGHT * np.maximum(0, (semantic - SEMANTIC_FLOOR) / (1 - SEMANTIC_FLOOR))
                  + LEXICAL_WEIGHT * lexical)
        candidates = [index for index, chunk in enumerate(self.chunks)
                      if self._eligible(chunk, context, normalized)]
        ranked = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)[:k]
        if not ranked or float(scores[ranked[0]]) < MIN_SCORE:
            return {**context, "status": "no_evidence", "message": "Não encontrei trechos suficientes para responder. Consulte a fonte oficial ou o setor responsável.", "sources": []}
        selected = [{**self.chunks[i], "score": round(float(scores[i]), 5),
                     "semantic_score": round(float(semantic[i]), 5)} for i in ranked]
        return {**context, "sources": selected}


def retrieve(question, semester=None, get_retriever=Retriever):
    """Aplica os filtros de escopo antes de carregar o modelo de busca.

    `get_retriever` permite reutilizar uma instância (por exemplo, em cache no Streamlit).
    """
    context = question_context(question, semester)
    if context["status"] != "ok":
        return {**context, "sources": []}
    return get_retriever().search(question, semester, context=context)

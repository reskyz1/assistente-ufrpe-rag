"""Extrai artigos e linhas de calendário sem separar as datas dos eventos."""

import csv
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "dados" / "fontes.csv"
CHUNKS = ROOT / "dados" / "trechos.json"
# Semestres cobertos pelo calendário da base. Atualize ao incluir outro calendário.
SUPPORTED_SEMESTERS = ("2026.1", "2026.2")
ARTICLE = re.compile(r"^\s*Art\.\s*(\d+)\s*[.º°]", re.I)
MONTH = re.compile(
    r"^\s*(Janeiro|Fevereiro|Março|Abril|Maio|Junho|Julho|Agosto|"
    r"Setembro|Outubro|Novembro|Dezembro)(?:\s*/\s*(\d{4}))?\s*$", re.I
)
DATE_ROW = re.compile(
    r"^\s*(\d{1,2}(?:/\d{2})?(?:\s+(?:a|e)\s+\d{1,2}(?:/\d{2})?)?)\s{2,}(\S.*)$"
)


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))


def clean(text):
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def sources():
    with SOURCES.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def download_sources(force=False):
    # O truststore usa a cadeia de certificados do Windows, incluindo intermediários.
    if os.name == "nt":
        import truststore
        truststore.inject_into_ssl()
    for source in sources():
        target = ROOT / source["local_path"]
        expected = source["sha256"]
        if not force and target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"PDF alterado: {target.name}. Revise a fonte antes de reindexar.")
            continue
        response = requests.get(source["pdf_url"], timeout=(15, 90))
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"A fonte {source['document_id']} não retornou um PDF.")
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError("A fonte mudou. Revise o documento e atualize o SHA-256 no manifesto.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.pdf.tmp")
        temporary.write_bytes(response.content)
        temporary.replace(target)


def useful_lines(page):
    for line in (page.extract_text(extraction_mode="layout") or "").splitlines():
        value = line.strip()
        if not value or any(marker in value for marker in (
            "MINISTÉRIO DA EDUCAÇÃO", "UNIVERSIDADE FEDERAL RURAL",
            "SECRETARIA GERAL DOS CONSELHOS", "CONSELHO DE ENSINO",
            "Confere com o original", "(ANEXO DA RESOLUÇÃO", "(CONTINUAÇÃO DA RESOLUÇÃO",
        )):
            continue
        yield line


def record(source, identifier, text, pages, **metadata):
    return {
        "id": identifier, "document_id": source["document_id"],
        "title": source["title"], "url": source["pdf_url"],
        "page_url": source["page_url"], "pages": sorted(set(pages)),
        "checked_at": source["checked_at"], "text": text, **metadata,
    }


def article_chunks(source, reader):
    articles = []
    current = None
    lower, upper = int(source["article_min"]), int(source["article_max"])
    for page_number, page in enumerate(reader.pages, start=1):
        for line in useful_lines(page):
            match = ARTICLE.match(line)
            if match:
                if current:
                    articles.append(current)
                number = int(match.group(1))
                current = {"article": number, "lines": [], "pages": []} if lower <= number <= upper else None
            if current and not re.match(r"\s*(CAPÍTULO|SEÇÃO|SUBSEÇÃO)\b", line, re.I):
                current["lines"].append(line)
                current["pages"].append(page_number)
    if current:
        articles.append(current)
    result = []
    for article in articles:
        # Remove o título do capítulo seguinte, que aparece após o último parágrafo.
        lines = article["lines"]
        while lines and lines[-1].strip().isupper():
            lines.pop()
        words = clean("\n".join(lines)).split()
        for part, start in enumerate(range(0, len(words), 145), start=1):
            text = " ".join(words[start:start + 180])
            if start:
                text = f"Art. {article['article']} (continuação). {text}"
            result.append(record(source, f"rgg-art{article['article']}-{part}", text,
                                 article["pages"], article=article["article"],
                                 kind="regulation", semester="", event=""))
            if start + 180 >= len(words):
                break
    return result


def calendar_chunks(source, reader):
    rows = []
    current = None
    month, year = "", source["year"]
    for page_number, page in enumerate(reader.pages, start=1):
        if page_number < 3:
            continue
        for line in useful_lines(page):
            match_month = MONTH.match(line)
            match_date = DATE_ROW.match(line)
            if match_month:
                if current:
                    rows.append(current)
                    current = None
                month = match_month.group(1).capitalize()
                year = match_month.group(2) or year
            elif match_date:
                if current:
                    rows.append(current)
                current = {"date": clean(match_date.group(1)), "month": month,
                           "year": year, "lines": [match_date.group(2)], "pages": [page_number]}
            elif current and not re.search(r"PERÍODO LETIVO DE|EVENTOS ACADÊMICOS", line):
                current["lines"].append(line)
                current["pages"].append(page_number)
    if current:
        rows.append(current)
    result = []
    for index, row in enumerate(rows, start=1):
        description = clean("\n".join(row["lines"]))
        normalized = normalize(description)
        if not re.search(r"matricula|trancamento|suspensao de programa", normalized):
            continue
        # Linhas de ajuste de vagas podem ter descrição antes da coluna de datas.
        # O recorte desta versão usa apenas eventos com início completo reconhecível.
        if not re.match(r"(?:data limite|periodo|solicitacao|processamento|matricula|analise)\b", normalized):
            continue
        semesters = sorted(set(re.findall(r"20\d{2}\.[12]", description)))
        # A primeira versão cobre somente os dois semestres da graduação presencial de 2026.
        if len(semesters) != 1 or semesters[0] not in SUPPORTED_SEMESTERS:
            continue
        event = "matricula"
        for candidate in ("trancamento", "suspensao", "rematricula", "extraordinaria"):
            if candidate in normalized:
                event = candidate
                break
        text = f"{row['month']}/{row['year']} - {row['date']}. {description}"
        result.append(record(source, f"cal2026-{index}", text, row["pages"],
                             kind="calendar", article=None, semester=semesters[0], event=event))
    return result


def build_corpus():
    download_sources()
    chunks = []
    for source in sources():
        reader = PdfReader(ROOT / source["local_path"])
        extractor = article_chunks if source["kind"] == "regulation" else calendar_chunks
        extracted = extractor(source, reader)
        if not extracted:
            raise ValueError(f"Nenhum trecho extraído de {source['document_id']}.")
        chunks.extend(extracted)
    CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHUNKS.with_suffix(f".{os.getpid()}.json.tmp")
    temporary.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(CHUNKS)
    return chunks


def load_corpus():
    if not CHUNKS.exists():
        raise FileNotFoundError("Execute python -m rag_ufrpe preparar antes da consulta.")
    return json.loads(CHUNKS.read_text(encoding="utf-8"))

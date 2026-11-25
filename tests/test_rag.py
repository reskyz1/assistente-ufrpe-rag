"""Regressões de extração, filtros de pergunta, busca, avaliação e falhas de geração."""

import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from rag_ufrpe.corpus import ROOT, SUPPORTED_SEMESTERS, load_corpus, sources
from rag_ufrpe.evaluation import evaluate_retrieval, load_cases, rank_metrics
from rag_ufrpe.generation import (answer, calendar_answer, numbers_in, strip_fence,
                                  unsupported_share)
from rag_ufrpe.retrieval import Retriever, calendar_matches, question_context, retrieve


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_corpus()

    def test_pdfs_match_manifest(self):
        for source in sources():
            digest = hashlib.sha256((ROOT / source["local_path"]).read_bytes()).hexdigest()
            self.assertEqual(digest, source["sha256"], source["document_id"])

    def test_ids_are_unique(self):
        ids = [chunk["id"] for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_article_of_the_section_is_present(self):
        articles = {c["article"] for c in self.chunks if c["kind"] == "regulation"}
        self.assertEqual(articles, set(range(229, 265)))

    def test_calendar_events_cover_only_supported_semesters(self):
        semesters = {c["semester"] for c in self.chunks if c["kind"] == "calendar"}
        self.assertEqual(semesters, set(SUPPORTED_SEMESTERS))

    def test_article_continues_on_next_page(self):
        article = next(c for c in self.chunks if c["id"] == "rgg-art248-1")
        self.assertIn("§ 4º", article["text"])
        self.assertEqual(article["pages"], [56, 57])

    def test_calendar_month_survives_page_break(self):
        event = next(c for c in self.chunks if c["id"] == "cal2026-107")
        self.assertTrue(event["text"].startswith("Agosto/2026 - 10 a 28."))
        self.assertEqual(event["semester"], "2026.2")
        self.assertIn(12, event["pages"])

    def test_calendar_first_semester_is_distinct(self):
        event = next(c for c in self.chunks if c["id"] == "cal2026-36")
        self.assertTrue(event["text"].startswith("Março/2026 - 09 a 28."))
        self.assertEqual(event["semester"], "2026.1")


class QuestionTests(unittest.TestCase):
    def test_date_requires_semester(self):
        self.assertEqual(question_context("Qual o prazo de matrícula?")["status"], "clarify")

    def test_plural_dates_use_calendar(self):
        self.assertTrue(question_context("Quais as datas de matrícula em 2026.1?")["date_question"])

    def test_unsupported_year(self):
        self.assertEqual(question_context("Qual o prazo de matrícula em 2027.1?")["status"], "unsupported_period")

    def test_conflicting_semesters(self):
        self.assertEqual(question_context("Qual o prazo de matrícula em 2026.1?", "2026.2")["status"], "clarify")

    def test_out_of_scope(self):
        self.assertEqual(question_context("Qual é a capital da França?")["status"], "out_of_scope")
        self.assertEqual(question_context("Como faço um bolo de chocolate?")["status"], "out_of_scope")

    def test_empty_question(self):
        self.assertEqual(question_context("   ")["status"], "empty")

    def test_general_rule_does_not_require_semester(self):
        self.assertEqual(question_context("O que significa trancamento de disciplina?")["status"], "ok")

    def test_alternative_semester_formats(self):
        for text in ("2026-1", "2026/1", "2026.1", "1º semestre de 2026"):
            context = question_context(f"Qual o prazo de trancamento em {text}?")
            self.assertEqual((context["status"], context["semester"]), ("ok", "2026.1"), text)

    def test_alternative_format_of_unsupported_semester(self):
        self.assertEqual(question_context("Qual o prazo de matrícula em 2027/1?")["status"], "unsupported_period")

    def test_deadline_word_in_a_rule_question_is_not_a_date_question(self):
        for text in ("Qual o prazo máximo para o trancamento ser efetivado?",
                     "Posso pedir suspensão de programa depois do prazo por motivo de trabalho?"):
            context = question_context(text)
            self.assertEqual((context["status"], context["date_question"]), ("ok", False), text)

    def test_data_limite_is_a_date_question(self):
        context = question_context("Qual a data limite para pedir REMT em 2026.1?")
        self.assertEqual((context["status"], context["date_question"]), ("ok", True))

    def test_rule_vocabulary_without_the_main_keywords_stays_in_scope(self):
        self.assertEqual(question_context("Uma turma optativa pode ser cancelada se tiver poucos alunos?")["status"], "ok")

    def test_discipline_deadline_excludes_program_suspension(self):
        self.assertFalse(calendar_matches({"event": "suspensao"}, "prazo de trancamento de disciplina"))

    def test_regular_enrollment_excludes_extraordinary(self):
        self.assertFalse(calendar_matches({"event": "extraordinaria", "text": "Matrícula extraordinária"}, "quando e a matricula"))

    def test_reenrollment_defaults_to_the_student_request(self):
        request = {"event": "rematricula", "text": "Solicitação de rematrícula dos alunos regulares"}
        processing = {"event": "rematricula", "text": "Processamento das solicitações de rematrícula"}
        self.assertTrue(calendar_matches(request, "quando e a rematricula"))
        self.assertFalse(calendar_matches(processing, "quando e a rematricula"))
        self.assertTrue(calendar_matches(processing, "quando e o processamento da rematricula"))


class FakeEncoder:
    """Codificador determinístico por contagem de palavras, para testar sem baixar o modelo."""

    def encode(self, texts, **_):
        vectors = np.zeros((len(texts), 64))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[row, hash(word.strip(".,?:")) % 64] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


def make_chunk(identifier, text, kind="regulation", semester="", event=""):
    return {"id": identifier, "kind": kind, "semester": semester, "event": event, "text": text,
            "title": "Teste", "url": "u", "pages": [1]}


CHUNKS = [
    make_chunk("art-1", "Art. 1. O trancamento de disciplina é efetivado um dia após a solicitação."),
    make_chunk("art-2", "Art. 2. A suspensão de programa tem limite de quatro períodos."),
    make_chunk("cal-1", "Março/2026 - 09 a 28. Período de trancamento de disciplina 2026.1", "calendar", "2026.1", "trancamento"),
    make_chunk("cal-2", "Agosto/2026 - 10 a 28. Período de trancamento de disciplina 2026.2", "calendar", "2026.2", "trancamento"),
    make_chunk("cal-3", "Março/2026 - 02/03 a 29/05. Período de suspensão de programa 2026.1", "calendar", "2026.1", "suspensao"),
]


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(CHUNKS, encoder=FakeEncoder())

    def ids(self, question, semester=None, k=3):
        return [item["id"] for item in self.retriever.search(question, semester, k)["sources"]]

    def test_date_question_sees_only_calendar_of_the_semester(self):
        self.assertEqual(self.ids("Qual o prazo de trancamento de disciplina em 2026.2?"), ["cal-2"])

    def test_date_question_uses_event_type(self):
        self.assertEqual(self.ids("Quando posso pedir suspensão de programa em 2026.1?"), ["cal-3"])

    def test_rule_question_never_returns_calendar_events(self):
        found = self.ids("Em quanto tempo o trancamento de disciplina é efetivado?")
        self.assertTrue(found and all(item.startswith("art-") for item in found))

    def test_k_limits_the_results(self):
        self.assertEqual(len(self.ids("trancamento de disciplina efetivado suspensão", k=1)), 1)

    def test_k_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search("trancamento", k=0)

    def test_abstention_returns_no_sources(self):
        found = self.retriever.search("Qual a capital da França?")
        self.assertEqual((found["status"], found["sources"]), ("out_of_scope", []))

    def test_missing_calendar_event_is_no_evidence(self):
        found = self.retriever.search("Quando é a rematrícula em 2026.1?")
        self.assertEqual((found["status"], found["sources"]), ("no_evidence", []))

    def test_retrieve_skips_loading_the_model_when_abstaining(self):
        def forbidden():
            raise AssertionError("o modelo não deveria ser carregado")
        self.assertEqual(retrieve("Qual a capital da França?", get_retriever=forbidden)["status"], "out_of_scope")

    def test_retrieve_passes_the_context_through(self):
        found = retrieve("Qual o prazo de trancamento em 2026.1?", get_retriever=lambda: self.retriever)
        self.assertEqual([item["id"] for item in found["sources"]], ["cal-1"])


class EvaluationTests(unittest.TestCase):
    def test_rank_metrics_first_hit_in_second_place(self):
        metrics = rank_metrics(["x", "a", "y"], {"a"})
        self.assertEqual(metrics["reciprocal_rank"], .5)
        self.assertFalse(metrics["hit_at_1"])
        self.assertTrue(metrics["hit_at_3"])
        self.assertAlmostEqual(metrics["precision_at_3"], 1 / 3)

    def test_precision_returned_ignores_missing_positions(self):
        metrics = rank_metrics(["a"], {"a"})
        self.assertAlmostEqual(metrics["precision_at_3"], 1 / 3)
        self.assertEqual(metrics["precision_returned"], 1.0)

    def test_rank_metrics_without_results(self):
        metrics = rank_metrics([], {"a"})
        self.assertEqual((metrics["reciprocal_rank"], metrics["precision_returned"], metrics["f1_at_3"]), (0.0, 0.0, 0.0))

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            load_cases("inexistente")

    def test_all_labels_point_to_existing_chunks(self):
        known = {chunk["id"] for chunk in load_corpus()}
        for case in load_cases("all"):
            for identifier in filter(None, case["expected_ids"].split(";")):
                self.assertIn(identifier, known, case["id"])

    def test_evaluation_runs_without_writing_files(self):
        class Stub:
            chunks = load_corpus()

            def search(self, question, semester=None, k=3):
                return {"status": "out_of_scope", "sources": []}

        results = ROOT / "avaliacao" / "resultados_dev.csv"
        before = results.stat().st_mtime_ns
        summary = evaluate_retrieval("dev", Stub(), write=False)
        self.assertEqual(results.stat().st_mtime_ns, before)
        self.assertEqual(summary["hit_at_1"], 0.0)
        self.assertEqual(summary["answerable_cases"], 6)

    def test_label_for_unknown_chunk_is_rejected(self):
        class Stub:
            chunks = [{"id": "outro"}]

        with self.assertRaises(ValueError):
            evaluate_retrieval("dev", Stub(), write=False)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.retrieval = {"status": "ok", "message": "", "sources": [{"id": "test-1", "title": "Teste", "pages": [1], "text": "Regra da fonte."}]}

    @patch("rag_ufrpe.generation.local_generate", return_value="Uma resposta sem fonte.")
    def test_missing_citation_is_rejected(self, generate):
        self.assertEqual(answer(self.retrieval, "Matrícula?")["mode"], "invalid_citations")

    @patch("rag_ufrpe.generation.local_generate", return_value='{"resposta":"Uma resposta.","fontes":[99]}')
    def test_unknown_citation_is_rejected(self, generate):
        self.assertEqual(answer(self.retrieval, "Matrícula?")["mode"], "invalid_citations")

    @patch("rag_ufrpe.generation.local_generate", return_value='{"resposta":"Regra da fonte.","fontes":[1]}')
    def test_structured_citation_is_rendered(self, generate):
        result = answer(self.retrieval, "Matrícula?")
        self.assertEqual(result["mode"], "generated")
        self.assertEqual(result["answer"], "Regra da fonte. [1]")

    @patch("rag_ufrpe.generation.local_generate", return_value='```json\n{"resposta":"Regra da fonte.","fontes":[1]}\n```')
    def test_fenced_json_is_accepted(self, generate):
        self.assertEqual(answer(self.retrieval, "Matrícula?")["mode"], "generated")

    @patch("rag_ufrpe.generation.local_generate", return_value='{"resposta":"O trancamento é imediato.","fontes":[1]}')
    def test_time_contradiction_shows_original_rule(self, generate):
        self.retrieval["sources"][0]["text"] = "O trancamento é efetivado 1 (um) dia após a solicitação."
        result = answer(self.retrieval, "Em quanto tempo o trancamento é efetivado?")
        self.assertEqual(result["mode"], "grounding_warning")
        self.assertIn("1 (um) dia", result["answer"])

    @patch("rag_ufrpe.generation.local_generate", return_value='{"resposta":"O limite é 8 períodos.","fontes":[1]}')
    def test_number_absent_from_source_is_blocked(self, generate):
        self.retrieval["sources"][0]["text"] = "O limite é de 4 períodos."
        self.assertEqual(answer(self.retrieval, "Qual o limite de suspensão?")["mode"], "grounding_warning")

    @patch("rag_ufrpe.generation.local_generate", return_value='{"resposta":"Sim, você pode solicitar.","fontes":[1]}')
    def test_affirmative_answer_to_prohibition_shows_conditions(self, generate):
        self.retrieval["sources"][0]["text"] = "Não pode ser solicitada suspensão. Há exceções por motivo de saúde."
        result = answer(self.retrieval, "Posso solicitar suspensão?")
        self.assertEqual(result["mode"], "grounding_warning")
        self.assertIn("exceções por motivo de saúde", result["answer"])

    @patch("rag_ufrpe.generation.local_generate",
           return_value='{"resposta":"Não precisa estar quite com a biblioteca.","fontes":[1]}')
    def test_negative_answer_without_negation_in_source_shows_original_rule(self, generate):
        self.retrieval["sources"][0]["text"] = "A suspensão só é realizada se comprovada a quitação com a biblioteca."
        result = answer(self.retrieval, "Preciso estar quite com a biblioteca?")
        self.assertEqual(result["mode"], "grounding_warning")
        self.assertIn("quitação", result["answer"])

    @patch("rag_ufrpe.generation.local_generate",
           return_value='{"resposta":"Não é permitido trancar em período de férias.","fontes":[1]}')
    def test_negative_answer_supported_by_the_source_is_kept(self, generate):
        self.retrieval["sources"][0]["text"] = "Não é permitido o trancamento em período letivo especial de férias."
        self.assertEqual(answer(self.retrieval, "Posso trancar nas férias?")["mode"], "generated")

    @patch("rag_ufrpe.generation.local_generate",
           return_value='{"resposta":"A carga mínima é definida pelo Código de Processo Penal, com 40 horas.","fontes":[1]}')
    def test_invented_content_shows_original_rule(self, generate):
        self.retrieval["sources"][0]["text"] = "Art. 232. A carga horária máxima semanal é de 40 (quarenta) horas."
        result = answer(self.retrieval, "Qual a carga horária mínima do curso?")
        self.assertEqual(result["mode"], "grounding_warning")

    @patch("rag_ufrpe.generation.local_generate",
           return_value='{"resposta":"A fonte não traz informação sobre isso.","fontes":[]}')
    def test_explicit_abstention_is_not_an_invalid_citation(self, generate):
        result = answer(self.retrieval, "Quanto custa a matrícula?")
        self.assertEqual(result["mode"], "abstention")
        self.assertEqual(len(result["sources"]), 1)

    @patch("rag_ufrpe.generation.local_generate", side_effect=OSError("offline"))
    def test_generation_failure_keeps_sources(self, generate):
        result = answer(self.retrieval, "Matrícula?")
        self.assertEqual(result["mode"], "generation_error")
        self.assertEqual(len(result["sources"]), 1)

    @patch("rag_ufrpe.generation.local_generate", side_effect=ImportError("sem transformers"))
    def test_missing_dependency_keeps_sources(self, generate):
        self.assertEqual(answer(self.retrieval, "Matrícula?")["mode"], "generation_error")

    @patch("rag_ufrpe.generation.local_generate")
    def test_abstention_does_not_call_model(self, generate):
        result = answer({"status": "out_of_scope", "message": "Fora do tema.", "sources": []}, "Capital?")
        self.assertEqual(result["mode"], "abstention")
        generate.assert_not_called()

    @patch("rag_ufrpe.generation.local_generate")
    def test_calendar_answer_does_not_call_model(self, generate):
        event = {"id": "cal-1", "kind": "calendar", "pages": [1],
                 "text": "Agosto/2026 - 10 e 11. Solicitação de rematrícula dos alunos regulares."}
        result = answer({"status": "ok", "message": "", "sources": [event]}, "Quando é a rematrícula?")
        self.assertEqual(result["mode"], "calendar")
        self.assertEqual(result["answer"], "Solicitação de rematrícula dos alunos regulares: Agosto/2026 - 10 e 11. [1]")
        generate.assert_not_called()

    def test_calendar_answer_format(self):
        self.assertEqual(calendar_answer({"text": "Março/2026 - 02. Data limite."}), "Data limite: Março/2026 - 02. [1]")

    def test_numbers_include_words_but_not_articles(self):
        self.assertEqual(numbers_in("4 (quatro) períodos e duas vezes, uma só"), {4, 2})

    def test_number_written_in_words_is_supported_by_digits(self):
        self.retrieval["sources"][0]["text"] = "O limite é de 4 períodos."
        with patch("rag_ufrpe.generation.local_generate", return_value='{"resposta":"O limite é de quatro períodos.","fontes":[1]}'):
            self.assertEqual(answer(self.retrieval, "Qual o limite?")["mode"], "generated")

    def test_unsupported_share(self):
        source = "O trancamento de matrícula é efetivado um dia após a solicitação."
        self.assertEqual(unsupported_share("O trancamento é efetivado um dia após a solicitação.", source, ""), 0.0)
        self.assertGreater(unsupported_share("Segundo o Código Penal, aplica-se multa pesada.", source, ""), .45)

    def test_strip_fence(self):
        self.assertEqual(strip_fence('```json\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(strip_fence('{"a": 1}'), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()

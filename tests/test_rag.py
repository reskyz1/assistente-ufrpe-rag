"""Regressões de extração entre páginas, filtro temporal e falhas de geração."""

import unittest
from unittest.mock import patch

from rag_ufrpe.corpus import load_corpus
from rag_ufrpe.generation import answer
from rag_ufrpe.retrieval import question_context, calendar_matches


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_corpus()

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

    def test_general_rule_does_not_require_semester(self):
        self.assertEqual(question_context("O que significa trancamento de disciplina?")["status"], "ok")

    def test_discipline_deadline_excludes_program_suspension(self):
        self.assertFalse(calendar_matches({"event": "suspensao"}, "prazo de trancamento de disciplina"))

    def test_regular_enrollment_excludes_extraordinary(self):
        self.assertFalse(calendar_matches({"event": "extraordinaria", "text": "Matrícula extraordinária"}, "quando e a matricula"))


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

    @patch("rag_ufrpe.generation.local_generate", side_effect=OSError("offline"))
    def test_generation_failure_keeps_sources(self, generate):
        result = answer(self.retrieval, "Matrícula?")
        self.assertEqual(result["mode"], "generation_error")
        self.assertEqual(len(result["sources"]), 1)

    @patch("rag_ufrpe.generation.local_generate")
    def test_abstention_does_not_call_model(self, generate):
        result = answer({"status": "out_of_scope", "message": "Fora do tema.", "sources": []}, "Capital?")
        self.assertEqual(result["mode"], "abstention")
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

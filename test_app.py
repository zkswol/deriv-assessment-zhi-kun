"""Focused pipeline checks; no API calls or credentials required."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import app


class FileInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_missing_empty_and_unsupported_document_directories(self):
        with self.assertRaises(ValueError):
            app.load_documents(self.root / "missing")
        for filename, text in [("empty.md", " \n"), ("ignored.csv", "policy")]:
            (self.root / filename).write_text(text, encoding="utf-8")
        with self.assertRaises(ValueError):
            app.load_documents(self.root)

    def test_nested_utf8_documents_with_same_basename_keep_paths(self):
        for folder in ["a", "b"]:
            (self.root / folder).mkdir()
            (self.root / folder / "policy.MD").write_text("Café support policy.", encoding="utf-8")
        docs = app.load_documents(self.root)
        self.assertEqual([doc_id for doc_id, _ in docs], ["a/policy.MD", "b/policy.MD"])
        self.assertTrue(all(text == "Café support policy." for _, text in docs))

    def test_invalid_question_shapes_and_duplicate_ids(self):
        cases = [{}, [], [None], [{"id": "q1"}], [{"id": 1, "question": "hello"}],
                 [{"id": " ", "question": "hello"}], [{"id": "q1", "question": " "}],
                 [{"id": "q1", "question": "one"}, {"id": "q1", "question": "two"}]]
        path = self.root / "questions.json"
        for data in cases:
            with self.subTest(data=data):
                app.save_json(path, data)
                with self.assertRaises(ValueError):
                    app.load_questions(path)

    def test_malformed_json_is_rejected(self):
        path = self.root / "questions.json"
        path.write_text("[{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            app.load_questions(path)


class ChunkTests(unittest.TestCase):
    def test_invalid_chunk_sizes_and_overlap(self):
        for size, overlap in [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                app.chunk_documents([("doc.md", "Some text")], size, overlap)

    def test_empty_and_heading_only_documents_are_rejected(self):
        for documents in [[], [("empty.md", " \n")], [("heading.md", "# Title\n\n## Subtitle")]]:
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                app.chunk_documents(documents)

    def test_exact_boundary_and_zero_overlap(self):
        words = [f"word{n}" for n in range(120)]
        self.assertEqual(len(app.chunk_documents([("doc.md", " ".join(words))])), 1)
        chunks = app.chunk_documents([("doc.md", " ".join(words + ["last"]))], overlap=0)
        self.assertEqual([len(chunk["text"].split()) for chunk in chunks], [120, 1])
        self.assertEqual(chunks[-1]["text"], "last")

    def test_windows_blank_lines_split_paragraphs(self):
        chunks = app.chunk_documents([("doc.md", "First policy.\r\n\r\nSecond policy.")])
        self.assertEqual(len(chunks), 2)

    def test_paragraphs_preserve_text_filename_and_unique_ids(self):
        chunks = app.chunk_documents([
            ("policies/access.md", "# Access\n\nReset links expire.\n\nAccounts can be locked."),
            ("other.txt", "Contact support."),
        ])
        self.assertEqual([chunk["text"] for chunk in chunks],
                         ["Reset links expire.", "Accounts can be locked.", "Contact support."])
        self.assertEqual([chunk["doc_id"] for chunk in chunks],
                         ["policies/access.md", "policies/access.md", "other.txt"])
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))

    def test_long_passage_preserves_all_words_and_overlap(self):
        words = [f"word{number}" for number in range(275)]
        chunks = app.chunk_documents([("long.md", " ".join(words))])
        self.assertTrue(all(len(chunk["text"].split()) <= 120 for chunk in chunks))
        reconstructed = chunks[0]["text"].split()
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous["text"].split()[-20:], current["text"].split()[:20])
            reconstructed.extend(current["text"].split()[20:])
        self.assertEqual(reconstructed, words)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = app.BM25Index(app.chunk_documents(app.load_documents(app.ROOT / "docs")))

    def test_expected_source_ranks_first(self):
        cases = [
            ("password resets per hour", "auth_policy.md"),
            ("API rate limit per token", "api_limits.md"),
            ("identity verification requires", "identity_verification.md"),
            ("approved bank withdrawals", "withdrawals.md"),
            ("service outage affecting all users", "incidents.md"),
        ]
        for question, expected_doc in cases:
            with self.subTest(question=question):
                retrieved = self.index.retrieve(question)
                self.assertEqual(retrieved[0]["doc_id"], expected_doc)
                self.assertLessEqual(len(retrieved), 3)
                self.assertTrue(all(chunk["score"] > 0 for chunk in retrieved))

    def test_empty_index_rejected(self):
        with self.assertRaises(ValueError):
            app.BM25Index([])

    def test_invalid_questions_and_top_k(self):
        for question in [None, 123, "", " \n"]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                self.index.retrieve(question)
        for top_k in [0, -1, 1.5, True, "3"]:
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.index.retrieve("password", top_k)

    def test_punctuation_stopwords_and_case(self):
        for question in ["???", "the and of"]:
            self.assertEqual(self.index.retrieve(question), [])
        self.assertEqual(self.index.retrieve("PASSWORD RESET!"), self.index.retrieve("password reset"))

    def test_single_chunk_and_stable_tie_order(self):
        chunks = app.chunk_documents([("b.md", "Reset password quickly."), ("a.md", "Reset password safely.")])
        index = app.BM25Index(chunks)
        self.assertEqual([row["doc_id"] for row in index.retrieve("password")], ["a.md", "b.md"])
        single = app.BM25Index(chunks[:1]).retrieve("password", top_k=10)
        self.assertEqual(len(single), 1)
        self.assertGreater(single[0]["score"], 0)

    def test_no_matches_produce_refusal_without_model_call(self):
        question = "Which cryptocurrencies will be added next year?"
        retrieved = self.index.retrieve(question)
        self.assertEqual(retrieved, [])
        with patch.object(app, "generate_gemini") as generate:
            result = app.answer_question(question, retrieved)
        generate.assert_not_called()
        self.assertFalse(result["supported"])
        self.assertEqual(result["answer"], app.REFUSAL)
        self.assertEqual(result["citations"], [])

    def test_exact_duplicates_do_not_fill_retrieval_slots(self):
        passage = "Users can request up to 3 password resets per hour."
        index = app.BM25Index(app.chunk_documents([
            ("auth.md", passage), ("faq.md", passage),
            ("links.md", "A password reset link expires after 15 minutes."),
        ]))
        retrieved = index.retrieve("password reset resets", top_k=2)
        self.assertEqual(len(retrieved), 2)
        self.assertEqual(len({chunk["text"] for chunk in retrieved}), 2)
        self.assertIn("links.md", {chunk["doc_id"] for chunk in retrieved})

    def test_different_limits_and_partial_overlap_are_preserved(self):
        passages = [
            ("standard.md", "API rate limit is 60 requests per minute."),
            ("premium.md", "API rate limit is 120 requests per minute."),
            ("retry.md", "API rate limit is 60 requests per minute. Retry after a pause."),
        ]
        index = app.BM25Index(app.chunk_documents(passages))
        retrieved = index.retrieve("API rate limit")
        self.assertEqual({chunk["doc_id"] for chunk in retrieved}, {doc_id for doc_id, _ in passages})


class AnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = app.BM25Index(app.chunk_documents(app.load_documents(app.ROOT / "docs")))

    def test_sample_answer_and_refusal_behaviors(self):
        for item in app.load_questions(app.ROOT / "questions.json"):
            with self.subTest(question=item["id"]):
                chunks = self.index.retrieve(item["question"])
                result = app.answer_question(item["question"], chunks, "extractive")
                self.assertEqual(result["supported"], item["expected_behavior"] == "answerable")
                self.assertEqual(app.validate_answer(result, chunks), [])

    def test_new_question(self):
        question = "When does a password reset link expire?"
        result = app.answer_question(question, self.index.retrieve(question), "extractive")
        self.assertTrue(result["supported"])
        self.assertIn("15 minutes", result["answer"])

    def test_fabricated_citation_and_quote_rejected(self):
        chunks = self.index.retrieve("password reset")
        for doc_id, quote in [("missing.md", "Invented"), ("auth_policy.md", "Unlimited resets")]:
            result = {"answer": "Unlimited resets", "supported": True, "citations": [doc_id],
                      "evidence": [{"doc_id": doc_id, "quote": quote}]}
            self.assertTrue(app.validate_answer(result, chunks))

    def test_invalid_model_answer_becomes_refusal(self):
        with patch.dict(app.os.environ, {"GEMINI_API_KEY": "test-placeholder"}), patch.object(
                app, "generate_gemini", return_value={"answer": "Invented", "supported": True,
                "citations": ["missing.md"], "evidence": [], "generator": "gemini"}):
            result = app.answer_question("password reset", self.index.retrieve("password reset"))
        self.assertFalse(result["supported"])
        self.assertEqual(result["answer"], app.REFUSAL)
        self.assertEqual(result["reason"], "validation_failed")

    def test_api_failure_has_labelled_fallback(self):
        question = "What is the API rate limit per API token?"
        with patch.dict(app.os.environ, {"GEMINI_API_KEY": "test-placeholder"}), patch.object(
                app, "generate_gemini", side_effect=TimeoutError):
            result = app.answer_question(question, self.index.retrieve(question))
        self.assertEqual(result["generator"], "extractive")
        self.assertEqual(result["fallback_reason"], "TimeoutError")
        self.assertTrue(result["supported"])

    def test_missing_key_does_not_call_provider(self):
        question = "What is the API rate limit per API token?"
        with patch.dict(app.os.environ, {}, clear=True), patch.object(app, "generate_gemini") as generate:
            result = app.answer_question(question, self.index.retrieve(question))
        generate.assert_not_called()
        self.assertEqual(result["fallback_reason"], "missing_api_key")

    def test_partial_question_and_incorrect_number_refused(self):
        for question in ["What is the API rate limit and monthly price?",
                         "Can users request 999 password resets per hour?"]:
            with self.subTest(question=question):
                result = app.answer_question(question, self.index.retrieve(question), "extractive")
                self.assertFalse(result["supported"])
                self.assertEqual(result["citations"], [])

    def test_invalid_answer_shapes_and_empty_answers(self):
        valid = {"answer": "Policy", "supported": True, "citations": ["auth_policy.md"], "evidence": []}
        for result in [None, [], {}, dict(valid, answer=" "), dict(valid, supported="true"),
                       dict(valid, citations="auth_policy.md"), dict(valid, citations=[123]),
                       dict(valid, citations=[]), dict(valid, evidence=None)]:
            with self.subTest(result=result):
                self.assertTrue(app.validate_answer(result, self.index.chunks))

    def test_quote_must_belong_to_its_cited_document(self):
        quote = "The API rate limit is 60 requests per minute per API token."
        result = {"answer": quote, "supported": True, "citations": ["auth_policy.md"],
                  "evidence": [{"doc_id": "auth_policy.md", "quote": quote}]}
        self.assertTrue(app.validate_answer(result, self.index.chunks))

    def test_each_citation_needs_its_own_evidence(self):
        result = app.generate_extractive("password resets per hour", self.index.chunks)
        result["citations"].append("api_limits.md")
        self.assertTrue(app.validate_answer(result, self.index.chunks))

    def test_unsupported_model_claim_is_replaced_with_standard_refusal(self):
        with patch.dict(app.os.environ, {"GEMINI_API_KEY": "placeholder"}), patch.object(
                app, "generate_gemini", return_value={"answer": "Maybe unlimited resets.", "supported": False,
                "citations": ["auth_policy.md"], "evidence": [], "generator": "gemini"}):
            result = app.answer_question("password reset", self.index.retrieve("password reset"))
        self.assertEqual(result["answer"], app.REFUSAL)
        self.assertEqual(result["citations"], [])

    def test_provider_error_message_is_not_logged(self):
        with patch.dict(app.os.environ, {"GEMINI_API_KEY": "placeholder"}), patch.object(
                app, "generate_gemini", side_effect=RuntimeError("sensitive-error-message")), patch.object(
                app, "log_stage") as log:
            result = app.answer_question("password resets per hour", self.index.retrieve("password resets"))
        self.assertEqual(result["fallback_reason"], "RuntimeError")
        self.assertNotIn("sensitive-error-message", str(log.call_args_list))


class SavedAnswerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(app, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.answer = app.refusal("extractive")

    def test_ids_follow_highest_number_and_repeated_question_updates(self):
        app.save_json(self.root / "answers.json", [app.answer_record("q8", self.answer), app.answer_record("q2", self.answer)])
        self.assertEqual(app.save_cli_answer("first", self.answer), "q9")
        self.assertEqual(app.save_cli_answer("second", self.answer), "q10")
        self.assertEqual(app.save_cli_answer("first", self.answer), "q9")
        rows = json.loads((self.root / "answers.json").read_text())
        self.assertEqual([row["question_id"] for row in rows], ["q2", "q8", "q9", "q10"])

    def test_corrupt_answers_are_not_overwritten(self):
        path = self.root / "answers.json"
        for contents in ["broken json", "{}", "[null]"]:
            path.write_text(contents, encoding="utf-8")
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                app.save_cli_answer("new question", self.answer)
            self.assertEqual(path.read_text(), contents)

    def test_batch_reset_prevents_stale_cli_id_overwriting_evaluation(self):
        self.assertEqual(app.save_cli_answer("old question", self.answer), "q1")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "policy.md").write_text("Password links expire after 15 minutes.")
        app.save_json(self.root / "questions.json", [{"id": "q1", "question": "password links expire"}])
        with patch.object(app.sys, "argv", ["app.py", "--generator", "extractive"]), patch.object(
                app, "configure_logging"), patch("builtins.print"):
            app.main()
        rows = json.loads((self.root / "answers.json").read_text())
        original = rows[0].copy()
        self.assertEqual(app.save_cli_answer("old question", self.answer), "q2")
        self.assertEqual(json.loads((self.root / "answers.json").read_text())[0], original)


if __name__ == "__main__":
    unittest.main()

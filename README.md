# Local Support Assistant

A small Python CLI that answers support questions using local documents.
It uses BM25 to find relevant passages and Gemini to write an answer with
document citations. If the documents do not support the question, it refuses.
The sample policies are fictional, not official Deriv policies.

## Architecture and files

The flow is: **read documents → split into chunks → retrieve → build prompt →
generate answer → validate → save results**.

These stages are separate functions in one source file.

| File | Purpose |
| --- | --- |
| `app.py` | Pipeline and command-line interface |
| `docs/` | Local Markdown and text documents |
| `questions.json` | Evaluation questions and expected behavior |
| `retrieval_results.json` | Retrieved passages, source IDs, and BM25 scores |
| `answers.json` | Answers, citations, and supported flags |
| `validation_report.json` | Validation pass/fail results and errors |
| `test_app.py` | Offline tests for normal and edge cases |
| `DESIGN.md` | Design notes and decisions |
| `requirements.txt` | Dependencies for Gemini mode |
| `pipeline.log` | Generated JSON stage logs; ignored by Git |

## Setup

Use Python 3.10 or newer. Tested with Python 3.14.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Gemini API key to `.env`. `GEMINI_MODEL` selects the model; the default
is `gemini-3.5-flash-lite`. Keep `.env` private: it is ignored by Git.

## Run

Run all evaluation questions and regenerate the three output JSON files:

```powershell
python app.py
```

Ask a new question:

```powershell
python app.py --question "When does a password reset link expire?"
```

The CLI returns the answer, citations, supported flag, and retrieved sources.
It saves the answer and validation result using the next question ID. Repeating
a question updates its record. Full evaluation replaces these with batch results.

Run offline without a key or external packages:

```powershell
python app.py --generator extractive
```

Offline mode copies sentences from documents. Missing keys or API failures
also trigger it. Validation records identify the generator. Committed outputs
were generated offline.

Test retrieval only, without calling Gemini:

```powershell
python app.py --retrieve-only
```

This regenerates only `retrieval_results.json`. Replace files in `docs/` and
`questions.json` to test other inputs. Outputs are saved beside `app.py`.

Run the 33 offline tests:

```powershell
python -m unittest test_app -v
```

## How retrieval works

Documents are split at blank lines. Long paragraphs become chunks of up to
120 words, with 20 words of overlap to preserve nearby context. Each chunk
keeps its filename and a chunk ID.

BM25 ranks passages by matching words, word rarity, and passage length.
Exact duplicate passages are removed before selecting the top three positive
matches, leaving room for distinct evidence. The index is rebuilt each run.
Scores measure relevance, not answer confidence.

## Grounding and refusal

Gemini receives only the question and retrieved passages as knowledge. The
prompt asks for a concise answer, source filenames, and exact evidence quotes.
It must refuse if any part of the question is unsupported. Retrieved content
is treated as data, not instructions.

Code checks nonempty answers, boolean support flags, valid source citations,
and exact evidence quotes. Failed checks produce a refusal without citations.
`validation_report.json` records pass/fail and errors. A correct refusal passes
validation with `supported: false`.

## Limitations and tradeoffs

- A CLI and in-memory index keep setup small; there is no web UI or database.
- Keyword search is fast and local, but may miss differently worded questions.
- Gemini writes natural answers but needs network access and may make mistakes.
  Requests have a 20-second timeout.
- Citation and quote checks do not prove that every generated claim is correct.
- Offline answers preserve source wording but can refuse valid paraphrases.
- Exact duplicate filtering keeps one source for identical text. It does not
  remove near duplicates or resolve conflicting policies.
- Logs exclude credentials and document text. Log rotation and concurrent
  writes are outside this MVP's scope.

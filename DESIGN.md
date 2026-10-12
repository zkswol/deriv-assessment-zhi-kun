# Design notes: local support assistant

## What we are building
A small command-line assistant that answers support questions by searching
local product documentation. Each answer names its source document. When the
documents do not contain enough information, it says so instead of guessing.

Example: a user asks how many password resets are allowed per hour. The
assistant finds the password policy and returns its answer with that filename.
If the user asks about an undocumented VIP exception, it refuses to invent one.

## Inputs and outputs
- `docs/`: at least five short Markdown or text documents.
- `questions.json`: at least eight questions, including five answerable and
  three unsupported or partially answerable questions.
- `retrieval_results.json`: passages found for each question, with filenames,
  chunk IDs, relevance scores, and text.
- `answers.json`: answers, document citations, and a supported flag.
- `validation_report.json`: results of the checks on each answer.

The sample policies are fictional, not official Deriv policies. The evaluator
can replace the documents and questions; the code must not hardcode answers.

## How a question moves through the system
1. **Read:** load local `.md` and `.txt` files using UTF-8.
2. **Split:** break documents into short passages called chunks. Keep the
   filename and a stable ID on every chunk.
3. **Search:** build a searchable index and rank passages against the question.
4. **Prepare context:** put the question and selected passages into a separate
   generation input, including instructions to use only the supplied evidence.
5. **Answer:** return a concise supported answer and citations, or a refusal.
6. **Check:** validate the answer and citations using explicit code rules.

Retrieval, prompt construction, generation, and validation remain separate
functions in `app.py`, even though they share one file.

## Minimal architecture and decisions
Use one Python source file and a CLI. Retrieval uses the standard library;
Gemini generation uses `google-genai` and `.env` loading uses `python-dotenv`.
No database or server is required. An offline fallback needs no external API.

- **Trade-off:** a CLI is quick to run and test, but it does not provide a web UI.
- **Edge case:** reject empty questions and clearly report missing or invalid
  input files.

Use BM25, a keyword-based ranking method, to retrieve the top three passages
with positive scores. It gives more weight to useful, less common words.
Build the index in memory each time the program starts.

- **Trade-off:** no embedding model download or API cost; paraphrases may be missed.
- **Edge case:** a high search score means related text, not proof of an answer.

Use Gemini as the default generator. Supply only the question and retrieved
passages, request structured answers and verbatim evidence, and instruct it to
refuse questions that are not fully supported. Load credentials from environment
variables. Validate citations and evidence before returning an answer.
On API failure or a missing key, use an explicitly labelled extractive fallback
with a conservative question-coverage rule. `--generator extractive` selects it
directly. The generation functions remain separate from retrieval and validation.

- **Trade-off:** Gemini handles paraphrases but adds latency, costs, and grounding
  risks. The fallback runs without credentials but may refuse valid paraphrases.
- **Edge case:** a question with an unsupported extra request should receive
  a clear refusal rather than a confidently incomplete answer.

## Grounding and validation rules
- Answer text must not be empty.
- The supported flag must be a boolean.
- Every supported answer must include at least one citation.
- Every cited filename must appear in the retrieved passages.
- Unsupported answers must explicitly state that the documents are insufficient.
- For the extractive baseline, answer evidence must come verbatim from the
  cited retrieved text.

These checks establish clear boundaries. They do not prove every answer is
semantically correct. Retrieved document content is evidence, never permission
to change system instructions. Any later external API credentials must come
from environment variables and remain outside Git.

## Interface and delivery plan
Planned commands:
- `python app.py`: run all evaluation questions and regenerate the three artifacts.
- `python app.py --question "..."`: return an answer, citations, support flag,
  and retrieved source IDs for a new question.

Four sequential milestones:
1. Design notes, sample inputs, ingestion, chunking, and retrieval.
2. Grounded generation, refusal behavior, and individual-question CLI.
3. Validation, focused tests, and structured pipeline logs.
4. Regenerate artifacts, write README, inspect `.gitignore`, and push to GitHub.

Target: 35–40 minutes for implementation and checks, keeping the final 10
minutes for documentation and submission. The bounded improvement is the
exact duplicate chunk filter described below. Prioritize required artifacts before extras.

## Practical improvement: exact duplicate chunk filtering
After scoring passages, remove identical full-text chunks before taking the
top three results. This prevents repeated policy text from occupying limited
context slots, allowing more distinct evidence without additional API calls.
Keep the highest-ranked occurrence and its source metadata. Ties use chunk ID
order. Retrieval logs include the number of duplicate candidates removed.

- **Trade-off:** only one source is retained for repeated text; formatting
  differences and near duplicates are not removed.
- **Edge case:** partial overlap and different policy numbers remain intact.
  The existing 20-word chunk overlap is unchanged.

## Acceptance checks
From a clean checkout, an evaluator can run the main flow, regenerate all
outputs, and ask a new question. Tests should check chunk metadata, known
retrieval results, unsupported-question refusal, and invalid citations.
Stage logs should record document loading, chunking, retrieval, generation,
and validation without recording credentials.

## Observability
`pipeline.log` appends one JSON object per line, with a UTC timestamp and run ID.
It records document/chunk counts, retrieval counts and duration, generation
mode and duration, fallback reasons, validation outcomes, and run completion.
Logs exclude question text, document contents, answers, and credentials.
The local log is ignored by Git; log rotation is deferred for this small MVP.

## Edge-case test coverage
Run `python -m unittest test_app -v` for 33 offline tests. They cover missing or
empty document folders, nested UTF-8 files, malformed questions and duplicate
IDs, chunk boundaries and overlap, invalid retrieval inputs, exact duplicates,
partial questions, malformed answers, evidence from the wrong source, missing
keys and provider failures, privacy of error logs, corrupt saved files, and
sequential CLI IDs after batch regeneration. Temporary folders isolate test
writes from the real documents and answer artifacts. Provider calls are mocked;
these tests do not establish live Gemini's semantic accuracy.

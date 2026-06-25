# Superdoc Quickstart

You need Python 3.13, Docker Desktop, and Git. That's it.

---

## 1. Clone and enter the repo

```bash
git clone <repo-url>
cd nexus-superdoc-bot
```

---

## 2. Drop in the credentials

You should have received two files — put them in the project root:

```
nexus-superdoc-bot/
├── .env
└── credentials.json
```

Don't rename them. The app looks for them exactly there.

---

## 3. Set up a Python virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Mac / Linux
source venv/bin/activate

# Windows (CMD)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

---

## 4. Build the dev Docker image

```bash
python sd.py build
```

This only needs to run once (or after requirements.txt changes).

---

## 5. Run the tests

```bash
# All tests
python sd.py test

# Just the merge pipeline
python sd.py test src/tests/test_merge.py

# Just tree tests (no Google Docs calls, faster)
python sd.py test src/tests/test_tree.py

# One specific test
python sd.py test src/tests/test_merge.py::TestFullMerge::test_full_merge_basic_text
```

The first run will automatically create Google Docs for each test PDF and save their IDs to `src/tests/test_docs.json`. Subsequent runs reuse those docs.

---

## 6. Drop into the container shell (optional)

If you want to poke around inside the container:

```bash
python sd.py shell
```

---

## Common issues

**`Error: No such file or directory: .env`**
The .env file is missing from the project root. Make sure you put it there, not inside `src/`.

**`ModuleNotFoundError: No module named 'src'`**
You're running pytest directly instead of through `sd.py`. Always use `python sd.py test` — it sets `PYTHONPATH` correctly inside the container.

**All tests show `s` (skipped)**
Your PDFs aren't where the test runner expects them. Put your test PDFs in a `files/` folder at the project root:
```
nexus-superdoc-bot/
└── files/
    ├── basic-text.pdf
    └── 2301.07041v2.pdf
```

**`docker: command not found`**
Docker Desktop isn't installed or isn't running. Start Docker Desktop and try again.

**`Exception: Heading already exists` on heading tests**
A previous test run failed mid-way and left a heading in the Google Doc. Run the test again — the tests clean up stale headings on startup.

**Google auth errors (`invalid_grant`, token errors)**
The credentials in `.env` may have expired. Ask whoever gave you the credentials for a fresh `.env`.

---

## Project layout (so you know where things are)

```
nexus-superdoc-bot/
├── sd.py                   ← dev runner, run this for everything
├── Dockerfile.dev          ← dev container (not for Lambda)
├── Dockerfile              ← Lambda deployment image, don't touch
├── requirements.txt
├── .env                    ← you put this here
├── credentials.json        ← you put this here
├── files/                  ← put your test PDFs here
└── src/
    ├── superdoc.py         ← main orchestrator, start reading here
    ├── core/
    │   ├── semantic_renderer.py   ← PDF → EmbedTreeNode tree
    │   ├── merge_algs.py          ← TreeEmbedder + SemanticReconciler
    │   └── gdocs_renderer.py      ← EmbedTreeNode → Docs API requests
    ├── models/
    │   └── tree_nodes.py          ← EmbedTreeNode, GdocTreeNode
    ├── services/
    │   ├── openai_client.py       ← embeddings + LLM calls
    │   ├── pinecone_client.py     ← vector DB
    │   └── gdocs_client.py        ← Google Docs API
    └── tests/
        ├── conftest.py            ← fixtures, read this before writing tests
        ├── test_docs.json         ← auto-generated, maps PDFs to doc IDs
        ├── test_merge.py
        ├── test_tree.py
        ├── test_pinecone.py
        └── test_render.py
```

---

## Adding a test for a new PDF

1. Put the PDF in `files/`
2. Write a test using `make_superdoc`:

```python
def test_my_pdf(make_superdoc):
    sd, stream, doc_id = make_superdoc("my-lecture.pdf")
    sd.merge_pdf_hierarchical(stream=stream)
```

3. Run it — a Google Doc gets created automatically on first run and the ID is saved to `test_docs.json`.

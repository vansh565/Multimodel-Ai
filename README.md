# Trinetra AI: Local Multi-Domain Graph RAG

Ask questions about your own PDFs: Indian law (IPC), medical and disaster management. Everything runs on your machine. The PDFs are turned into a knowledge graph with semantic chunking, and answers come from a local **Mistral** model served by **Ollama**. No API keys and no cloud calls.

| Domain | Source folder |
|---|---|
| ⚖️ Legal & Constitutional | `data/IPC_docs/` |
| 🏥 Medical Information | `data/medical_docs/` |
| 🌊 Disaster Management | `data/disaster_docs/` |
| 🌐 All-in-One | every PDF under `data/` |

---

## How data flows

The app has two phases. **Indexing** runs once in the background when `app.py` starts. **Answering** runs for every question you send from the browser.

### 1. Indexing (startup)

![Animated diagram: PDF files become semantic chunks, then an entity graph, then communities, stored as an in-memory domain index](docs/indexing-flow.svg)

1. **PDF files → text.** `load_pdf()` in [app.py](app.py) reads each PDF in the domain's folder page by page with `pypdf`.
2. **Text → semantic chunks.** `semantic_chunk()` in [src/chunking/semantic_chunker.py](src/chunking/semantic_chunker.py) splits the text into sentences. It embeds each sentence together with its neighbours using `all-MiniLM-L6-v2`, then starts a new chunk wherever the similarity between consecutive sentences drops below **0.75**. Each chunk therefore holds one topic.
3. **Chunks → entity graph.** `build_graph()` in [src/graph/graph_builder.py](src/graph/graph_builder.py) runs spaCy NER (`en_core_web_sm`) on each chunk. Every named entity becomes a node, and entities found in the same chunk are linked by an edge that **stores the chunk text**.
4. **Graph → communities.** `detect_communities()` in [src/graph/community_detector.py](src/graph/community_detector.py) runs Louvain clustering, which groups entities that tend to appear together.
5. **Domain index.** The chunks, graph and communities are kept in memory in `system_states[domain]`. The domains are built in order: legal, medical, disaster, then all. Each PDF is chunked only once; the All-in-One index reuses the chunks already made.

Until a domain is ready, the UI shows *"Loading knowledge base…"* and questions to that domain are refused.

### 2. Answering a question (every request)

![Animated diagram: browser sends a question to Flask, which runs local and global search over the domain index, builds a prompt for Ollama, and streams tokens back to the browser while saving the chat](docs/question-flow.svg)

1. **Browser → Flask.** [templates/index.html](templates/index.html) sends `POST /api/ask/stream` with `{system, question, session_id}`.
2. **Memory commands.** `ask_stream()` in [app.py](app.py) first checks whether the message is a memory command (*"remember …"*, *"forget …"*, *"what do you remember?"*). Those are answered directly from `memory.db`, without the LLM. See [Memory](#memory-short-term--long-term).
3. **Short-term memory.** For a normal question, it loads the last 6 messages of the current chat. This is what lets follow-ups such as *"How is it treated?"* work.
4. **Local search.** `local_search()` in [src/retrieval/local_search.py](src/retrieval/local_search.py) embeds the question, together with the previous question so follow-ups work, and returns up to 6 passages:
   - **Graph (top 3):** entities with similarity above **0.5** score the chunks on their edges. Each entity's score is weighted by how specific it is (IDF), so *"Section 420"*, which appears in 2 chunks, outranks *"IPC"*, which appears in 31.
   - **Direct (top 3):** the chunks most similar to the question itself. These catch passages whose entities are too generic to match, such as *"Symptoms"*.
5. **Global search.** `global_search()` in [src/retrieval/global_search.py](src/retrieval/global_search.py) compares the question with each community and returns the entity names of the **top 3 communities**. These names give the model broader topic context.
6. **Prompt.** `build_prompt()` in [src/llm/answer_generator.py](src/llm/answer_generator.py) combines the assistant's identity (*Trinetra AI, developed by Vansh Sharma*), the domain instructions, the retrieved passages and entities, **every long-term memory fact**, the chat history and the question.
7. **Local LLM.** `generate_stream()` in [src/llm/llm_client.py](src/llm/llm_client.py) sends the prompt to Ollama at `localhost:11434`, which generates the answer token by token.
8. **Streaming back.** Flask relays each token to the browser as a Server-Sent Event (`data: {"token": "…"}`) and finishes with `data: {"done": true, "session_id": …}`. The UI renders the answer as it arrives.
9. **Saving.** `save_exchange()` stores the question and answer in the chat session and writes them to `memory.db`. The first question becomes the chat's title in the sidebar.

### Memory: short-term + long-term

| | Short-term memory | Long-term memory |
|---|---|---|
| What | The last 6 messages of the current chat | Facts you told it to remember |
| Scope | One chat | Every chat, every domain |
| Survives restart | Yes (chats are saved to `memory.db`) | Yes (`memory.db`) |
| Code | `get_chat_context()` in [app.py](app.py) | [src/memory/memory_store.py](src/memory/memory_store.py) |

Teach it in the chat:

| You type | What happens |
|---|---|
| `remember my name is Arjun and I am a law student` | Saves the fact; every future answer can use it |
| `what do you remember?` | Lists all saved facts |
| `forget Arjun` | Deletes facts containing "Arjun" |
| `forget everything` | Clears long-term memory |

Questions such as *"do you remember my name?"* are **not** treated as commands; they go to the LLM, which answers from memory. The sidebar's **🧠 Long-term memory** panel shows every fact, and you can delete one with **×**.

The assistant's name and creator come from `ASSISTANT_NAME` and `ASSISTANT_CREATOR` (see [Configuration](#configuration)), so asking *"who developed you?"* answers *"Trinetra AI, developed by Vansh Sharma"* rather than *"Mistral AI"*.

---

## Quick start (Windows)

**Requirements:** Python 3.10+, [Ollama](https://ollama.com), and about 5 GB of disk space for the model.

```powershell
cd semrag_project

# 1. Python environment
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m spacy download en_core_web_sm

# 2. Local LLM (Ollama runs in the background after install)
winget install Ollama.Ollama
ollama pull mistral

# 3. Run
.venv\Scripts\python app.py
```

Open **http://127.0.0.1:5000**. All domains are ready about a minute after startup.

> **Using a shared global Python?** If you hit `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`, another package (e.g. `yolov5`) has pinned an old `huggingface_hub`. Create the venv with `python -m venv --system-site-packages .venv` and install only `"huggingface_hub>=1.5,<2"` into it. The venv then reuses your global torch while keeping the newer hub library separate.

## Deploy to Render (`app2.py`, Gemini API)

`app2.py` is the cloud version of `app.py`. It has the same domains, UI, memory and API. The differences:

| | `app.py` (local) | `app2.py` (cloud) |
|---|---|---|
| Answers | Ollama + Mistral on your PC | Google Gemini API (`gemini-3.5-flash-lite`) |
| Embeddings | `all-MiniLM-L6-v2` (PyTorch) | the same `all-MiniLM-L6-v2`, run with ONNX via `fastembed`; identical vectors, no PyTorch |
| Index | built at every startup (~1 min) | built once on your PC with `build_index.py`, loaded from `index_cache/` in about 2 seconds |
| Server packages | full `requirements.txt` (torch, spaCy, …) | `requirements-render.txt` (Flask, fastembed, numpy, …); uses about 200 MB, under Render's free 512 MB |
| Rate limit | none | 15 questions/min and 300/day per visitor (`ASK_RATE_LIMIT`) |

The chunks, entity graph, communities, embeddings and search are the **same** as in `app.py`. Only the answer-writing model changes. Gemini is used only for answers: about one API call per question, and none for "remember…" or "forget…" commands.

**1. Get a Gemini API key** at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and put it in `semrag_project/.env` (copy `.env.example`):

```
GEMINI_API_KEY=your-key
```

`.env` is listed in `.gitignore`. Never commit the key.

**2. Build the index locally** whenever the PDFs change:

```powershell
cd semrag_project
.venv\Scripts\python build_index.py
```

This takes under a minute and makes no API calls. It writes about 2 MB to `index_cache/`.

**3. Try it locally:** `.venv\Scripts\python app2.py`, then open http://127.0.0.1:5001.

**4. Commit and push** `app2.py`, `build_index.py`, `index_cache/`, `requirements-render.txt`, `src/cloud/` and `render.yaml` (in the repo root). The embedding model is not committed: Render downloads it into `.model_cache/` during the build.

**5. Create the service on Render:** **New → Blueprint**, pick this repository, and enter `GEMINI_API_KEY` when asked. Render reads the rest from [render.yaml](../render.yaml):

- Root directory `semrag_project`
- Build command `pip install -r requirements-render.txt`
- Start command `gunicorn app2:app --workers 1 --threads 8 --timeout 180 --bind 0.0.0.0:$PORT`

To set it up by hand instead (**New → Web Service**), use the same three values and add `GEMINI_API_KEY` under **Environment**.

> **On Render's free plan, the disk is wiped on every restart and redeploy.** Saved chats and long-term memory in `memory.db` reset then, and the free service also sleeps after 15 minutes without visitors. To keep memory, attach a Render persistent disk (paid plans) and set `MEMORY_DB` to a path on it.

## Adding your own PDFs

Put PDFs into the matching folder under `data/` and restart `app.py`. For `app2.py`, rerun `build_index.py` and push the new `index_cache/`. A PDF placed directly in `data/` (not in a domain folder) is included only in **All-in-One**. To add a new domain, add an entry to `DOMAINS` in [app.py](app.py) and a matching button in [templates/index.html](templates/index.html).

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `mistral` | Any model you have pulled, e.g. `llama3.2:3b` for a faster, smaller model |
| `DATA_DIR` | `./data` | Folder that holds the domain folders |
| `MEMORY_DB` | `./memory.db` | SQLite file for saved chats and long-term memory |
| `ASSISTANT_NAME` | `Trinetra AI` | Name the assistant uses for itself |
| `ASSISTANT_CREATOR` | `Vansh Sharma` | Who the assistant says developed it |
| `GEMINI_API_KEY` | (none) | `app2.py` only: Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | `app2.py` only: chat model |
| `ASK_RATE_LIMIT` | `15 per minute;300 per day` | `app2.py` only: questions allowed per visitor |
| `PORT` | `5001` | `app2.py` only: port (Render sets this) |

## API

| Method & path | Used for |
|---|---|
| `GET /` | The chat UI |
| `POST /api/ask/stream` | Ask a question; streams the answer as Server-Sent Events |
| `POST /api/ask` | Same, but returns JSON with the answer and the retrieved passages (answer cut to 25 words) |
| `GET /api/sessions/<system>` | List chats for a domain |
| `POST /api/sessions/<system>` | Start a new chat |
| `GET /api/sessions/<system>/<id>` | Messages of one chat |
| `DELETE /api/sessions/<system>/<id>` | Delete a chat |
| `GET /api/status/<system>` | Index status and PDFs for one domain |
| `GET /api/memory` | List long-term memory facts |
| `POST /api/memory` | Add a fact: `{"fact": "…"}` |
| `DELETE /api/memory/<id>` | Delete one fact |
| `GET /health` | Readiness of every domain and whether Ollama is reachable |

## Project layout

```
semrag_project/
├── app.py                     # Flask server: indexing, retrieval, streaming, chat sessions
├── app2.py                    # Same app for the cloud: Gemini API + prebuilt index (Render)
├── build_index.py             # Builds index_cache/ for app2.py (run locally)
├── index_cache/               # Prebuilt chunks, graph, communities + Gemini embeddings
├── requirements-render.txt    # Lightweight server dependencies for app2.py
├── .env.example               # Template for .env (GEMINI_API_KEY); .env itself is git-ignored
├── memory.db                  # Saved chats + long-term memory (created automatically)
├── templates/index.html       # Trinetra AI chat UI
├── data/                      # Your PDFs, one folder per domain
├── docs/                      # Animated diagrams used in this README
└── src/
    ├── chunking/semantic_chunker.py
    ├── graph/                 # entity_extractor.py, graph_builder.py, community_detector.py
    ├── retrieval/             # local_search.py, global_search.py
    ├── llm/                   # answer_generator.py (prompt), llm_client.py (Ollama)
    ├── memory/memory_store.py # long-term memory + chat persistence (SQLite)
    ├── cloud/                 # app2.py only: gemini_client.py, vector_index.py
    └── pipeline/              # older standalone scripts, not used by app.py
```

## Limitations

- **Memory is shared by everyone using this computer's server.** There are no user accounts, so all long-term facts apply to every chat.
- **The index is rebuilt on every start.** Nothing is cached to disk, so startup takes about a minute.
- **Answers are only as good as the retrieved passages.** The prompt asks for short answers, but the model sometimes writes longer ones, and passages about neighbouring topics can leak into an answer.

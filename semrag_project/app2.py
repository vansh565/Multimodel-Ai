"""Cloud version of app.py for deploying on Render (or any server).

Same features and API as app.py, but:
  * answers come from the Google Gemini API (GEMINI_API_KEY) instead of a local Ollama model
  * retrieval uses a prebuilt index (index_cache/, made locally with build_index.py) and the same
    all-MiniLM-L6-v2 embeddings run with ONNX (fastembed), so the server needs no PyTorch / spaCy
    and fits in a small instance
"""
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from pypdf import PdfReader
from src.cloud import embedder, gemini_client
from src.cloud.gemini_client import check_gemini_available
from src.cloud.vector_index import load_index, local_search_detailed, global_search, chunk_source
from src.llm.answer_generator import build_prompt
from src.memory import memory_store
import glob
import json
import os
import time
import traceback
import uuid
from datetime import datetime
from collections import defaultdict

# Resolve paths relative to this file so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(BASE_DIR, "index_cache"))
MEMORY_DB = os.getenv("MEMORY_DB", os.path.join(BASE_DIR, "memory.db"))

# Per-visitor limits so a public deployment can't burn through the free Gemini quota
ASK_RATE_LIMIT = os.getenv("ASK_RATE_LIMIT", "15 per minute;300 per day")

# Name the UI shows for the answer engine (the underlying model provider is not exposed to visitors)
ENGINE_NAME = os.getenv("ENGINE_NAME", "Trinetra Graph RAG")

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Trinetra AI")
ASSISTANT_CREATOR = os.getenv("ASSISTANT_CREATOR", "Vansh Sharma")
IDENTITY = (f"You are {ASSISTANT_NAME}, an AI assistant developed by {ASSISTANT_CREATOR}. "
            f"Only when asked who you are or who made, built or developed you, say you are {ASSISTANT_NAME}, "
            f"developed by {ASSISTANT_CREATOR}; otherwise do not introduce yourself. "
            f"Never say you were developed by Google, Gemini or any other company.")

# Knowledge domains shown in the UI. 'folder' is relative to DATA_DIR;
# 'all' (folder None) indexes every PDF under DATA_DIR, including ones in no domain folder.
DOMAINS = {
    'legal': {
        'name': 'Legal & Constitutional',
        'folder': 'IPC_docs',
        'instructions': 'You are a legal assistant for the Indian Constitution, IPC, cyber laws and legal governance.'
    },
    'medical': {
        'name': 'Medical Information',
        'folder': 'medical_docs',
        'instructions': 'You are a medical information assistant for diseases, treatments and healthcare. '
                        'Advise consulting a doctor for serious conditions.'
    },
    'disaster': {
        'name': 'Disaster Management',
        'folder': 'disaster_docs',
        'instructions': 'You are a disaster management assistant for natural disasters, emergency response, '
                        'NDRF/SDRF and safety protocols.'
    },
    'all': {
        'name': 'All-in-One',
        'folder': None,
        'instructions': 'You are an assistant for legal, medical and disaster management questions.'
    }
}

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
# Render sits behind a proxy: use X-Forwarded-For so rate limits apply per visitor, not per proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app)
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

# Per-domain index state
system_states = {
    key: {'ready': False, 'loading': False, 'error': None, 'status_message': 'Waiting...',
          'index': None, 'documents': []}
    for key in DOMAINS
}

# Long-term memory: remembered facts + saved chats (SQLite), so both survive restarts
memory_store.init(MEMORY_DB)

# chat_sessions[domain][session_id] = {'title', 'messages', 'created_at', 'last_updated'}
chat_sessions = defaultdict(dict, memory_store.load_sessions())

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        try:
            text += (page.extract_text() or "") + "\n"
        except Exception:
            pass
    return text

def find_pdfs(folder=None):
    root = os.path.join(DATA_DIR, folder) if folder else DATA_DIR
    return sorted(glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True))

def initialize_system():
    """Load the prebuilt index of every domain (takes seconds, so it runs at import time for gunicorn)"""
    for key in DOMAINS:
        state = system_states[key]
        try:
            index = load_index(INDEX_DIR, key)
            if index.get('embed_model') != embedder.MODEL_NAME:
                raise ValueError(f"index was built with {index.get('embed_model')}, "
                                 f"but the app uses {embedder.MODEL_NAME}; rerun build_index.py")
            state.update(index=index, documents=index['documents'], ready=True, status_message='Ready')
            print(f"[{key}] ready: {len(index['documents'])} PDFs, {len(index['chunks'])} chunks, "
                  f"{len(index['nodes'])} entities, {len(index['communities'])} communities", flush=True)
        except FileNotFoundError:
            state.update(error='Index not built', status_message='Index not built. Run: python build_index.py')
            print(f"[{key}] index missing in {INDEX_DIR}. Run: python build_index.py", flush=True)
        except Exception as e:
            traceback.print_exc()
            state.update(error=str(e), status_message=f'Initialization failed: {e}')

def get_chat_context(system, session_id, max_history=6):
    """Short-term memory: the last few messages of this chat, formatted for the prompt"""
    session = chat_sessions[system].get(session_id)
    if not session or not session['messages']:
        return ""

    context = "Previous conversation:\n"
    for msg in session['messages'][-max_history:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        context += f"{role}: {msg['content']}\n"
    return context

def get_or_create_session(system, session_id=None):
    if not session_id or session_id not in chat_sessions[system]:
        session_id = session_id or str(uuid.uuid4())
        chat_sessions[system][session_id] = {
            'title': 'New Chat',
            'messages': [],
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
    return session_id

def save_exchange(system, session_id, question, answer):
    session = chat_sessions[system][session_id]
    if not session['messages']:
        words = question.split()[:6]
        session['title'] = " ".join(words)[:40] or 'New Chat'
    now = datetime.now()
    session['messages'].append({'role': 'user', 'content': question, 'timestamp': now.isoformat()})
    session['messages'].append({'role': 'assistant', 'content': answer, 'timestamp': now.isoformat()})
    session['last_updated'] = now

    # Chats are only written to disk once they have messages, so empty "New Chat"s don't pile up
    memory_store.save_session(system, session_id, session['title'], session['created_at'], now)
    memory_store.add_message(session_id, 'user', question, now.isoformat())
    memory_store.add_message(session_id, 'assistant', answer, now.isoformat())

def instructions_for(system):
    return f"{IDENTITY}\n{DOMAINS[system]['instructions']}"

def remembered_facts():
    return [f['fact'] for f in memory_store.list_facts()]

def parse_ask_request():
    """Validate an ask request; returns (system, question, session_id, error)"""
    data = request.get_json(silent=True) or {}
    system = data.get('system', 'all')
    question = data.get('question', '').strip()
    if system not in DOMAINS:
        return None, None, None, f"Unknown system '{system}'"
    if not question:
        return None, None, None, 'No question provided'
    if not system_states[system]['ready']:
        return None, None, None, f"{DOMAINS[system]['name']} is not available: {system_states[system]['status_message']}"
    return system, question, get_or_create_session(system, data.get('session_id')), None

def search_query(system, session_id, question):
    """The question plus the previous one, so follow-ups like "How is it treated?" find the right passages.
    (The full chat history goes into the prompt; here it would only dilute the search.)"""
    session = chat_sessions[system].get(session_id)
    previous = next((m['content'] for m in reversed(session['messages']) if m['role'] == 'user'), '') if session else ''
    return f"{previous} {question}".strip()

def _clean(name):
    return " ".join(name.split())

def retrieve(system, session_id, question):
    """Returns (passages, topics, trace); the trace describes the retrieval for the UI's RAG panel"""
    started = time.time()
    index = system_states[system]['index']
    query_emb = embedder.embed([search_query(system, session_id, question)])[0]
    picks, entities = local_search_detailed(query_emb, index)
    topics = global_search(query_emb, index)

    session = chat_sessions[system].get(session_id)
    trace = {
        'mode': 'rag',
        'domain': DOMAINS[system]['name'],
        'graph_size': {'chunks': len(index['chunks']), 'entities': len(index['nodes'])},
        'entities': [{'name': name, 'similarity': round(sim, 2)}
                     for name, sim in list({_clean(n): s for n, s in entities}.items())[:6]],
        'passages': [{'source': chunk_source(index, p['chunk']) or DOMAINS[system]['name'],
                      'via': p['via'],
                      'entity': _clean(p['entity']) if p['entity'] else None,
                      'score': round(p['score'], 2),
                      'preview': _clean(index['chunks'][p['chunk']])[:280]} for p in picks],
        'topics': list(dict.fromkeys(_clean(t) for t in topics))[:8],
        'memory_facts': len(memory_store.list_facts()),
        'history_messages': min(6, len(session['messages'])) if session else 0,
        'retrieval_ms': round((time.time() - started) * 1000),
    }
    return [index['chunks'][p['chunk']] for p in picks], topics, trace

def build_answer_prompt(system, session_id, question, local_res, global_res):
    return build_prompt(question, local_res, global_res, instructions_for(system),
                        get_chat_context(system, session_id), remembered_facts())

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'error': f'Too many questions, please wait a moment ({e.description}).'}), 429

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'systems': {k: {'ready': s['ready'], 'error': s['error'], 'message': s['status_message']}
                    for k, s in system_states.items()},
        'llm_model': ENGINE_NAME,
        'llm_available': check_gemini_available()
    })

@app.route('/api/status/<system>')
def get_status(system):
    if system not in DOMAINS:
        return jsonify({'error': 'Unknown system'}), 404
    s = system_states[system]
    return jsonify({'ready': s['ready'], 'loading': s['loading'], 'error': s['error'],
                    'message': s['status_message'], 'documents': s['documents']})

@app.route('/api/sessions/<system>', methods=['GET'])
def list_sessions(system):
    if system not in DOMAINS:
        return jsonify({'error': 'Unknown system'}), 404
    sessions = []
    for sid, s in chat_sessions[system].items():
        last_user = next((m['content'] for m in reversed(s['messages']) if m['role'] == 'user'), '')
        sessions.append({
            'id': sid,
            'title': s['title'],
            'preview': last_user or 'No messages yet',
            'message_count': len(s['messages']),
            'last_updated': s['last_updated'].isoformat()
        })
    sessions.sort(key=lambda x: x['last_updated'], reverse=True)
    return jsonify({'sessions': sessions})

@app.route('/api/sessions/<system>', methods=['POST'])
def create_session(system):
    if system not in DOMAINS:
        return jsonify({'error': 'Unknown system'}), 404
    return jsonify({'session_id': get_or_create_session(system)})

@app.route('/api/sessions/<system>/<session_id>', methods=['GET'])
def get_session(system, session_id):
    session = chat_sessions[system].get(session_id) if system in DOMAINS else None
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'session_id': session_id, 'title': session['title'], 'messages': session['messages']})

@app.route('/api/sessions/<system>/<session_id>', methods=['DELETE'])
def delete_session(system, session_id):
    if system in DOMAINS and chat_sessions[system].pop(session_id, None):
        memory_store.delete_session(session_id)
        return jsonify({'success': True})
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/memory', methods=['GET'])
def list_memory():
    return jsonify({'facts': memory_store.list_facts()})

@app.route('/api/memory', methods=['POST'])
def add_memory():
    fact = ((request.get_json(silent=True) or {}).get('fact') or '').strip()
    if not fact:
        return jsonify({'error': 'No fact provided'}), 400
    return jsonify({'id': memory_store.add_fact(fact)})

@app.route('/api/memory/<int:fact_id>', methods=['DELETE'])
def delete_memory(fact_id):
    if memory_store.delete_fact(fact_id):
        return jsonify({'success': True})
    return jsonify({'error': 'Memory not found'}), 404

@app.route('/api/ask/stream', methods=['POST'])
@limiter.limit(ASK_RATE_LIMIT)
def ask_stream():
    """Server-sent events, in order:
    data: {"stage": "retrieving"}  data: {"rag": {...trace...}}  data: {"stage": "generating"}
    data: {"token": "..."} ...     data: {"done": true, "session_id": "..."}"""
    system, question, session_id, error = parse_ask_request()

    def sse(payload):
        return f"data: {json.dumps(payload)}\n\n"

    def stream():
        if error:
            yield sse({'error': error})
            return

        # "remember ...", "forget ...", "what do you remember" are answered from memory, without the LLM
        reply = memory_store.handle_command(question)
        if reply:
            yield sse({'rag': {'mode': 'memory', 'memory_facts': len(memory_store.list_facts())}})
            yield sse({'token': reply})
            save_exchange(system, session_id, question, reply)
            yield sse({'done': True, 'session_id': session_id, 'memory_changed': True})
            return

        yield sse({'stage': 'retrieving'})
        try:
            local_res, global_res, trace = retrieve(system, session_id, question)
        except Exception:
            traceback.print_exc()
            yield sse({'error': 'Searching the knowledge base failed. Please try again.'})
            return
        yield sse({'rag': trace})
        yield sse({'stage': 'generating'})

        answer = ""
        prompt = build_answer_prompt(system, session_id, question, local_res, global_res)
        for token in gemini_client.generate_stream(prompt):
            answer += token
            yield sse({'token': token})

        save_exchange(system, session_id, question, answer.strip())
        yield sse({'done': True, 'session_id': session_id})

    return Response(stream_with_context(stream()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/ask', methods=['POST'])
@limiter.limit(ASK_RATE_LIMIT)
def ask_question():
    system, question, session_id, error = parse_ask_request()
    if error:
        return jsonify({'error': error}), 400

    answer = memory_store.handle_command(question)
    local_res, global_res, trace = [], [], {'mode': 'memory'}
    if answer is None:
        local_res, global_res, trace = retrieve(system, session_id, question)
        response = gemini_client.generate(build_answer_prompt(system, session_id, question, local_res, global_res))
        answer = " ".join(response.split()[:25])  # same 25-word cut as app.py's generate_answer
    save_exchange(system, session_id, question, answer)

    return jsonify({
        'question': question,
        'answer': answer,
        'local_results': local_res,
        'global_results': global_res,
        'rag': trace,
        'system': system,
        'session_id': session_id
    })

# Runs on import so it also happens under gunicorn (Render), not only with "python app2.py"
initialize_system()
if any(s['ready'] for s in system_states.values()):
    embedder.get_model()  # load the embedding model now so the first question isn't slow
if not os.getenv("GEMINI_API_KEY"):
    print("WARNING: GEMINI_API_KEY is not set. Put it in .env locally, or in Render's Environment settings.")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    print(f"Open http://127.0.0.1:{port} in your browser", flush=True)
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False, threaded=True)

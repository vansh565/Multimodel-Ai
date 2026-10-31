from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from pypdf import PdfReader
from src.chunking.semantic_chunker import semantic_chunk
from src.graph.graph_builder import build_graph
from src.graph.community_detector import detect_communities
from src.retrieval.local_search import local_search
from src.retrieval.global_search import global_search
from src.llm.answer_generator import generate_answer, generate_answer_stream
from src.llm.llm_client import check_mistral_available, OLLAMA_MODEL
from src.memory import memory_store
import glob
import json
import os
import threading
import traceback
import uuid
from datetime import datetime
from collections import defaultdict

# Resolve paths relative to this file so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
MEMORY_DB = os.getenv("MEMORY_DB", os.path.join(BASE_DIR, "memory.db"))

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Trinetra AI")
ASSISTANT_CREATOR = os.getenv("ASSISTANT_CREATOR", "Vansh Sharma")
IDENTITY = (f"You are {ASSISTANT_NAME}, a local AI assistant developed by {ASSISTANT_CREATOR}. "
            f"Only when asked who you are or who made, built or developed you, say you are {ASSISTANT_NAME}, "
            f"developed by {ASSISTANT_CREATOR}; otherwise do not introduce yourself. "
            f"Never say you were developed by Mistral AI or any other company.")

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
CORS(app)

# Per-domain index state
system_states = {
    key: {'ready': False, 'loading': False, 'error': None, 'status_message': 'Waiting...',
          'graph': None, 'chunks': None, 'communities': None, 'documents': []}
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

def build_domain(key, pdf_paths, chunk_cache):
    state = system_states[key]
    state['loading'] = True
    state['documents'] = [os.path.relpath(p, DATA_DIR) for p in pdf_paths]
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found for '{key}' in {DATA_DIR}")

    # Chunk each PDF separately (and only once across domains) so chunks never span two documents
    chunks = []
    for i, path in enumerate(pdf_paths, 1):
        if path not in chunk_cache:
            state['status_message'] = f'Chunking {os.path.basename(path)} ({i}/{len(pdf_paths)})...'
            print(f"[{key}] {state['status_message']}", flush=True)
            chunk_cache[path] = semantic_chunk(load_pdf(path))
        chunks.extend(chunk_cache[path])

    state['status_message'] = 'Building graph...'
    graph = build_graph(chunks)
    state['status_message'] = 'Detecting communities...'
    communities = detect_communities(graph)

    state.update(graph=graph, chunks=chunks, communities=communities,
                 ready=True, loading=False, status_message='Ready')
    print(f"[{key}] ready: {len(pdf_paths)} PDFs, {len(chunks)} chunks, "
          f"{graph.number_of_nodes()} entities, {len(communities)} communities", flush=True)

def initialize_system():
    """Build every domain index in a background thread ('all' last, reusing cached chunks)"""
    chunk_cache = {}
    for key, domain in DOMAINS.items():
        try:
            build_domain(key, find_pdfs(domain['folder']), chunk_cache)
        except Exception as e:
            traceback.print_exc()
            system_states[key].update(loading=False, error=str(e),
                                      status_message=f'Initialization failed: {e}')

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
        return None, None, None, f"{DOMAINS[system]['name']} is still loading: {system_states[system]['status_message']}"
    return system, question, get_or_create_session(system, data.get('session_id')), None

def search_query(system, session_id, question):
    """The question plus the previous one, so follow-ups like "How is it treated?" find the right passages.
    (The full chat history goes into the prompt; here it would only dilute the search.)"""
    session = chat_sessions[system].get(session_id)
    previous = next((m['content'] for m in reversed(session['messages']) if m['role'] == 'user'), '') if session else ''
    return f"{previous} {question}".strip()

def retrieve(system, session_id, question):
    state = system_states[system]
    query = search_query(system, session_id, question)
    local_res = local_search(query, state['graph'], state['chunks'])
    global_res = global_search(query, state['communities'], state['chunks'])
    return local_res, global_res

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'systems': {k: {'ready': s['ready'], 'error': s['error'], 'message': s['status_message']}
                    for k, s in system_states.items()},
        'llm_model': OLLAMA_MODEL,
        'llm_available': check_mistral_available()
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
def ask_stream():
    """Server-sent events: data: {"token": ...} ... data: {"done": true}"""
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
            yield sse({'token': reply})
            save_exchange(system, session_id, question, reply)
            yield sse({'done': True, 'session_id': session_id, 'memory_changed': True})
            return

        history = get_chat_context(system, session_id)
        local_res, global_res = retrieve(system, session_id, question)

        answer = ""
        for token in generate_answer_stream(question, local_res, global_res,
                                            instructions_for(system), history, remembered_facts()):
            answer += token
            yield sse({'token': token})

        save_exchange(system, session_id, question, answer.strip())
        yield sse({'done': True, 'session_id': session_id})

    return Response(stream_with_context(stream()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    system, question, session_id, error = parse_ask_request()
    if error:
        return jsonify({'error': error}), 400

    answer = memory_store.handle_command(question)
    local_res, global_res = [], []
    if answer is None:
        history = get_chat_context(system, session_id)
        local_res, global_res = retrieve(system, session_id, question)
        answer = generate_answer(question, local_res, global_res, instructions_for(system),
                                 history, remembered_facts())
    save_exchange(system, session_id, question, answer)

    return jsonify({
        'question': question,
        'answer': answer,
        'local_results': local_res,
        'global_results': global_res,
        'system': system,
        'session_id': session_id
    })

if __name__ == "__main__":
    if not check_mistral_available():
        print(f"WARNING: Ollama model '{OLLAMA_MODEL}' not reachable at http://localhost:11434. "
              f"Install Ollama, then run 'ollama pull {OLLAMA_MODEL}'. Retrieval works, answers will show an error.")

    # Build the indexes in the background so the UI is available immediately
    init_thread = threading.Thread(target=initialize_system, daemon=True)
    init_thread.start()

    # use_reloader=False: the debug reloader would start a second process and rebuild the indexes twice
    print("Open http://127.0.0.1:5000 in your browser", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)

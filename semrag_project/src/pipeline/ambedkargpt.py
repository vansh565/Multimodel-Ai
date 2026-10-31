from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from src.chunking.semantic_chunker import semantic_chunk
from src.graph.graph_builder import build_graph
from src.graph.community_detector import detect_communities
from src.retrieval.local_search import local_search
from src.retrieval.global_search import global_search
from src.llm.answer_generator import generate_answer
import threading
import time

app = Flask(__name__)
CORS(app)

# Global variables to store system state
system_state = {
    'ready': False,
    'graph': None,
    'chunks': None,
    'communities': None,
    'loading': False,
    'status_message': ''
}

def initialize_system():
    """Initialize the system in a background thread"""
    system_state['loading'] = True
    system_state['status_message'] = 'Loading PDF...'
    time.sleep(0.5)
    
    text = load_pdf("data/diseases_real_200.pdf")
    
    system_state['status_message'] = 'Chunking...'
    time.sleep(0.5)
    chunks = semantic_chunk(text)
    system_state['chunks'] = chunks
    
    system_state['status_message'] = 'Building graph...'
    time.sleep(0.5)
    graph = build_graph(chunks)
    system_state['graph'] = graph
    
    system_state['status_message'] = 'Detecting communities...'
    time.sleep(0.5)
    communities = detect_communities(graph)
    system_state['communities'] = communities
    
    system_state['ready'] = True
    system_state['loading'] = False
    system_state['status_message'] = 'System Ready!'

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() + "\n"
        except:
            pass
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        'ready': system_state['ready'],
        'loading': system_state['loading'],
        'message': system_state['status_message']
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    if not system_state['ready']:
        return jsonify({'error': 'System not ready yet'}), 503
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Perform searches
    local_res = local_search(question, system_state['graph'], system_state['chunks'])
    global_res = global_search(question, system_state['communities'], system_state['chunks'])
    
    # Generate answer
    answer = generate_answer(question, local_res, global_res)
    
    return jsonify({
        'question': question,
        'answer': answer,
        'local_results': local_res,
        'global_results': global_res
    })

if __name__ == "__main__":
    # Start system initialization in background
    init_thread = threading.Thread(target=initialize_system)
    init_thread.start()
    
    # Run Flask app
    app.run(debug=True, port=5000)
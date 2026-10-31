from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from pypdf import PdfReader
from src.chunking.semantic_chunker import semantic_chunk
from src.graph.graph_builder import build_graph
from src.graph.community_detector import detect_communities
from src.retrieval.local_search import local_search
from src.retrieval.global_search import global_search
from src.llm.answer_generator import generate_answer
from collections import defaultdict
from datetime import datetime
import threading
import time
import os
import uuid
import json
import glob

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# System configurations
SYSTEMS = {
    'legal': {
        'name': 'Constitutional & Legal System',
        'icon': '⚖️',
        'color': '#1e3c72',
        'description': 'Indian Constitution, IPC, Cyber Laws, Legal Governance',
        'pdf_folder': 'data/IPC_docs',
        'pdf_patterns': ['*.pdf', '*.PDF'],
        'enabled': True
    },
    'medical': {
        'name': 'Medical Information System',
        'icon': '🏥',
        'color': '#2e7d32',
        'description': 'Diseases, Cures, Doctors, Treatments, Healthcare',
        'pdf_folder': 'data/medical_docs',
        'pdf_patterns': ['*.pdf', '*.PDF'],
        'enabled': True
    },
    'disaster': {
        'name': 'Disaster Management System',
        'icon': '🌊',
        'color': '#d32f2f',
        'description': 'Natural Disasters, Emergency Response, Safety Protocols',
        'pdf_folder': 'data/disaster_docs',
        'pdf_patterns': ['*.pdf', '*.PDF'],
        'enabled': True
    },
    'all': {
        'name': 'All-in-One Integrated System',
        'icon': '🌐',
        'color': '#9c27b0',
        'description': 'Combined knowledge from Legal, Medical & Disaster Management',
        'pdf_folder': None,
        'enabled': True
    }
}

# Store system states
system_states = {
    'legal': {'ready': False, 'graph': None, 'chunks': None, 'communities': None, 'loading': False, 'status_message': '', 'chunks_count': 0, 'pdf_files': []},
    'medical': {'ready': False, 'graph': None, 'chunks': None, 'communities': None, 'loading': False, 'status_message': '', 'chunks_count': 0, 'pdf_files': []},
    'disaster': {'ready': False, 'graph': None, 'chunks': None, 'communities': None, 'loading': False, 'status_message': '', 'chunks_count': 0, 'pdf_files': []},
    'all': {'ready': False, 'graph': None, 'chunks': None, 'communities': None, 'loading': False, 'status_message': '', 'chunks_count': 0, 'pdf_files': []}
}

# Chat sessions for each system
chat_sessions = defaultdict(lambda: defaultdict(lambda: {
    'messages': [],
    'created_at': datetime.now(),
    'last_updated': datetime.now(),
    'title': 'New Chat',
    'message_count': 0
}))

def find_pdf_files(folder_path):
    """Find all PDF files in a folder"""
    pdf_files = []
    if os.path.exists(folder_path):
        for pattern in ['*.pdf', '*.PDF']:
            pdf_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    return pdf_files

def load_pdf_text(pdf_path):
    """Extract text from PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except:
                continue
        return text
    except Exception as e:
        print(f"Error loading {pdf_path}: {str(e)}")
        return ""

def load_system_pdfs(system_key):
    """Load all PDFs for a specific system"""
    system_config = SYSTEMS[system_key]
    folder_path = system_config['pdf_folder']
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
        return None, []
    
    pdf_files = find_pdf_files(folder_path)
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return None, []
    
    print(f"Found {len(pdf_files)} PDF(s) in {folder_path}:")
    for pdf in pdf_files:
        print(f"  - {os.path.basename(pdf)}")
    
    # Load all PDFs
    combined_text = ""
    for pdf_file in pdf_files:
        text = load_pdf_text(pdf_file)
        if text:
            combined_text += f"\n\n--- Document: {os.path.basename(pdf_file)} ---\n\n"
            combined_text += text
    
    return combined_text if combined_text else None, pdf_files

def initialize_system(system_key):
    """Initialize a specific RAG system"""
    try:
        system_states[system_key]['loading'] = True
        system_states[system_key]['status_message'] = f'Loading {SYSTEMS[system_key]["name"]}...'
        print(f"\n{'='*50}")
        print(f"Initializing {SYSTEMS[system_key]['name']}...")
        print(f"{'='*50}")
        
        # Load PDFs
        text, pdf_files = load_system_pdfs(system_key)
        system_states[system_key]['pdf_files'] = pdf_files
        
        if not text:
            system_states[system_key]['status_message'] = f'No PDFs found in {SYSTEMS[system_key]["pdf_folder"]}'
            system_states[system_key]['ready'] = False
            system_states[system_key]['loading'] = False
            print(f"❌ No PDFs found for {SYSTEMS[system_key]['name']}")
            return
        
        print(f"📚 Loaded {len(pdf_files)} PDF(s), {len(text)} characters")
        system_states[system_key]['status_message'] = f'Chunking {SYSTEMS[system_key]["name"]}...'
        
        # Chunk the text
        chunks = semantic_chunk(text)
        system_states[system_key]['chunks'] = chunks
        system_states[system_key]['chunks_count'] = len(chunks)
        print(f"✅ Created {len(chunks)} chunks")
        
        system_states[system_key]['status_message'] = f'Building knowledge graph...'
        
        # Build graph
        graph = build_graph(chunks)
        system_states[system_key]['graph'] = graph
        print(f"✅ Graph built")
        
        system_states[system_key]['status_message'] = f'Detecting communities...'
        
        # Detect communities
        communities = detect_communities(graph)
        system_states[system_key]['communities'] = communities
        print(f"✅ Detected {len(communities)} communities")
        
        system_states[system_key]['ready'] = True
        system_states[system_key]['loading'] = False
        system_states[system_key]['status_message'] = f'{SYSTEMS[system_key]["name"]} Ready!'
        
        print(f"✅ {SYSTEMS[system_key]['name']} initialized successfully!")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"❌ Error initializing {SYSTEMS[system_key]['name']}: {str(e)}")
        system_states[system_key]['ready'] = False
        system_states[system_key]['loading'] = False
        system_states[system_key]['status_message'] = f'Error: {str(e)}'

def initialize_all_systems():
    """Initialize all systems in parallel"""
    threads = []
    
    for system_key in ['legal', 'medical', 'disaster']:
        if SYSTEMS[system_key]['enabled']:
            thread = threading.Thread(target=initialize_system, args=(system_key,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            time.sleep(0.5)  # Stagger startup
    
    # Wait for all to complete or timeout
    for thread in threads:
        thread.join(timeout=300)  # 5 minute timeout
    
    # Initialize ALL system (combines all)
    if system_states['legal']['ready'] or system_states['medical']['ready'] or system_states['disaster']['ready']:
        print("\n" + "="*50)
        print("Initializing All-in-One System...")
        print("="*50)
        
        combined_chunks = []
        if system_states['legal']['ready'] and system_states['legal']['chunks']:
            combined_chunks.extend(system_states['legal']['chunks'])
        if system_states['medical']['ready'] and system_states['medical']['chunks']:
            combined_chunks.extend(system_states['medical']['chunks'])
        if system_states['disaster']['ready'] and system_states['disaster']['chunks']:
            combined_chunks.extend(system_states['disaster']['chunks'])
        
        if combined_chunks:
            system_states['all']['chunks'] = combined_chunks
            system_states['all']['chunks_count'] = len(combined_chunks)
            system_states['all']['ready'] = True
            system_states['all']['status_message'] = f'All-in-One Ready! ({len(combined_chunks)} total chunks)'
            print(f"✅ All-in-One system ready with {len(combined_chunks)} total chunks")
        else:
            system_states['all']['ready'] = False
            system_states['all']['status_message'] = 'No systems available for All-in-One mode'

def generate_chat_title(messages, system_key):
    """Generate chat title based on system and content"""
    if not messages:
        return f"New {SYSTEMS[system_key]['name']} Chat"
    
    # Get first user message
    first_message = None
    for msg in messages:
        if msg['role'] == 'user':
            first_message = msg['content']
            break
    
    if not first_message:
        return f"New {SYSTEMS[system_key]['name']} Chat"
    
    # Extract first few words
    words = first_message.split()[:6]
    title = ' '.join(words)
    
    if len(title) > 40:
        title = title[:37] + "..."
    
    return f"{SYSTEMS[system_key]['icon']} {title.capitalize()}"

@app.route('/')
def index():
    return render_template('index.html', systems=SYSTEMS)

@app.route('/api/status/<system_key>')
def get_system_status(system_key):
    """Get status of a specific system"""
    if system_key not in system_states:
        return jsonify({'error': 'Invalid system'}), 404
    
    state = system_states[system_key]
    return jsonify({
        'ready': state['ready'],
        'loading': state['loading'],
        'message': state['status_message'],
        'chunks_count': state['chunks_count'],
        'pdf_files': [os.path.basename(f) for f in state.get('pdf_files', [])],
        'system_info': SYSTEMS[system_key]
    })

@app.route('/api/sessions/<system_key>', methods=['GET'])
def get_sessions(system_key):
    """Get all chat sessions for a system"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    sessions_list = []
    for session_id, session_data in chat_sessions[system_key].items():
        sessions_list.append({
            'id': session_id,
            'title': session_data.get('title', f"New {SYSTEMS[system_key]['name']} Chat"),
            'created_at': session_data['created_at'].isoformat(),
            'last_updated': session_data['last_updated'].isoformat(),
            'message_count': len(session_data['messages']),
            'preview': session_data['messages'][0]['content'][:50] + '...' if session_data['messages'] else 'Empty chat'
        })
    
    sessions_list.sort(key=lambda x: x['last_updated'], reverse=True)
    return jsonify({'sessions': sessions_list})

@app.route('/api/sessions/<system_key>', methods=['POST'])
def create_session(system_key):
    """Create a new chat session for a system"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    session_id = str(uuid.uuid4())
    chat_sessions[system_key][session_id] = {
        'messages': [],
        'created_at': datetime.now(),
        'last_updated': datetime.now(),
        'title': f"New {SYSTEMS[system_key]['name']} Chat",
        'message_count': 0
    }
    return jsonify({'session_id': session_id, 'title': chat_sessions[system_key][session_id]['title']})

@app.route('/api/sessions/<system_key>/<session_id>', methods=['GET'])
def get_session(system_key, session_id):
    """Get a specific chat session"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    if session_id not in chat_sessions[system_key]:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = chat_sessions[system_key][session_id]
    return jsonify({
        'id': session_id,
        'title': session_data.get('title', f"New {SYSTEMS[system_key]['name']} Chat"),
        'messages': session_data['messages'],
        'created_at': session_data['created_at'].isoformat(),
        'last_updated': session_data['last_updated'].isoformat(),
        'message_count': len(session_data['messages'])
    })

@app.route('/api/sessions/<system_key>/<session_id>', methods=['DELETE'])
def delete_session(system_key, session_id):
    """Delete a chat session"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    if session_id in chat_sessions[system_key]:
        del chat_sessions[system_key][session_id]
        return jsonify({'success': True})
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handle questions for any system"""
    data = request.json
    system_key = data.get('system', 'legal')
    question = data.get('question', '')
    session_id = data.get('session_id')
    
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 400
    
    if not system_states[system_key]['ready']:
        return jsonify({'error': f'{SYSTEMS[system_key]["name"]} is not ready yet'}), 503
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Store user message
    if session_id and session_id in chat_sessions[system_key]:
        chat_sessions[system_key][session_id]['messages'].append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        chat_sessions[system_key][session_id]['message_count'] += 1
        chat_sessions[system_key][session_id]['last_updated'] = datetime.now()
        
        # Update title
        new_title = generate_chat_title(chat_sessions[system_key][session_id]['messages'], system_key)
        chat_sessions[system_key][session_id]['title'] = new_title
    
    try:
        # Get system components
        graph = system_states[system_key]['graph']
        chunks = system_states[system_key]['chunks']
        communities = system_states[system_key]['communities']
        
        # Perform searches
        local_res = local_search(question, graph, chunks) if graph else []
        global_res = global_search(question, communities, chunks) if communities else []
        
        # Generate answer with system context
        context = f"You are an expert in {SYSTEMS[system_key]['name']}. {SYSTEMS[system_key]['description']}. "
        answer = generate_answer(question, local_res, global_res)
        
        # Add system-specific formatting
        if system_key == 'legal':
            answer = f"⚖️ **Legal Response:**\n\n{answer}"
        elif system_key == 'medical':
            answer = f"🏥 **Medical Information:**\n\n{answer}"
        elif system_key == 'disaster':
            answer = f"🌊 **Disaster Management:**\n\n{answer}"
        else:
            answer = f"🌐 **Integrated Response:**\n\n{answer}"
        
        # Store assistant response
        if session_id and session_id in chat_sessions[system_key]:
            chat_sessions[system_key][session_id]['messages'].append({
                'role': 'assistant',
                'content': answer,
                'timestamp': datetime.now().isoformat()
            })
            chat_sessions[system_key][session_id]['message_count'] += 1
            chat_sessions[system_key][session_id]['last_updated'] = datetime.now()
        
        return jsonify({
            'question': question,
            'answer': answer,
            'system': system_key,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/systems', methods=['GET'])
def get_systems():
    """Get all available systems and their status"""
    systems_status = {}
    for key in SYSTEMS:
        systems_status[key] = {
            'info': SYSTEMS[key],
            'status': {
                'ready': system_states[key]['ready'],
                'loading': system_states[key]['loading'],
                'chunks_count': system_states[key]['chunks_count']
            }
        }
    return jsonify(systems_status)

if __name__ == "__main__":
    # Create necessary folders
    for system in ['legal', 'medical', 'disaster']:
        folder = SYSTEMS[system]['pdf_folder']
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created folder: {folder}")
    
    # Start initialization in background
    init_thread = threading.Thread(target=initialize_all_systems)
    init_thread.daemon = True
    init_thread.start()
    
    print("\n" + "="*60)
    print("🚀 Multi-System RAG Platform Starting...")
    print("="*60)
    print("\n📂 Please place your PDFs in:")
    print("   - data/legal_docs/     (Constitution, IPC, Cyber Laws)")
    print("   - data/medical_docs/   (Diseases, Treatments, Doctors)")
    print("   - data/disaster_docs/  (Disaster Management, Safety)")
    print("\n🌐 Access the application at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
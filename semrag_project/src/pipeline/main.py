import os
import json
import uuid
import time
import logging
import secrets
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from logging.handlers import RotatingFileHandler

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Setup application logging"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(handler)
    
    return app_logger

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

limiter.init_app(app)

# Logger
logger = setup_logging()

# Mistral API Configuration
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', "mistral-small-latest")

if not MISTRAL_API_KEY:
    logger.error("MISTRAL_API_KEY not set in environment variables!")
    raise ValueError("MISTRAL_API_KEY is required")

# System configurations
SYSTEMS = {
    'legal': {
        'name': 'Constitutional & Legal System',
        'icon': '⚖️',
        'color': '#1e3c72',
        'description': 'Indian Constitution, IPC, Cyber Laws, Legal Governance',
        'pdf_folder': 'data/IPC_docs',
        'enabled': True
    },
    'medical': {
        'name': 'Medical Information System',
        'icon': '🏥',
        'color': '#2e7d32',
        'description': 'Diseases, Cures, Doctors, Treatments, Healthcare',
        'pdf_folder': 'data/medical_docs',
        'enabled': True
    },
    'disaster': {
        'name': 'Disaster Management System',
        'icon': '🌊',
        'color': '#d32f2f',
        'description': 'Natural Disasters, Emergency Response, Safety Protocols',
        'pdf_folder': 'data/disaster_docs',
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

SYSTEM_PROMPTS = {
    'legal': """
You are a legal expert specializing ONLY in:
- Indian Constitution
- IPC (Indian Penal Code)
- Cyber Laws
- Legal Governance

Rules:
1. Answer ONLY legal-domain questions.
2. If the question is outside legal topics, politely refuse.
3. Do NOT answer medical, disaster, coding, entertainment, politics, general knowledge, or unrelated questions.
4. Keep answers focused and concise.
5. Cite relevant Indian laws or articles when applicable.
6. Never generate fake legal information.

If a user asks out-of-domain questions, reply:
"I can only answer legal and constitutional questions."
""",

    'medical': """
You are a medical information expert specializing ONLY in:
- Diseases
- Treatments
- Healthcare
- Doctors
- Medicines
- Medical emergencies

Rules:
1. Answer ONLY medical and healthcare questions.
2. If the question is outside medical topics, politely refuse.
3. Do NOT answer legal, disaster, coding, entertainment, politics, or unrelated questions.
4. Keep answers focused and concise.
5. Always advise consulting healthcare professionals for serious conditions.
6. Never generate unsafe medical advice.

If a user asks out-of-domain questions, reply:
"I can only answer medical and healthcare questions."
""",

    'disaster': """
You are a disaster management expert specializing ONLY in:
- Natural disasters
- Emergency response
- Safety protocols
- Rescue operations
- Crisis management
- NDRF/SDRF safety guidance

Rules:
1. Answer ONLY disaster-management-related questions.
2. If the question is outside disaster management, politely refuse.
3. Do NOT answer legal, medical, coding, entertainment, politics, or unrelated questions.
4. Focus on practical safety measures.
5. Keep answers focused and concise.
6. Never generate unsafe emergency advice.

If a user asks out-of-domain questions, reply:
"I can only answer disaster management and safety related questions."
""",

    'all': """
You are an integrated expert in:
- Indian Constitution & IPC
- Medical Information
- Disaster Management

Rules:
1. Answer ONLY questions related to these three domains.
2. If the question is outside these domains, politely refuse.
3. Do NOT answer coding, entertainment, gaming, politics, unrelated general knowledge, or off-topic questions.
4. Keep answers accurate, concise, and domain-focused.
5. Never generate fake or misleading information.

If a user asks out-of-domain questions, reply:
"I can only answer legal, medical, and disaster-management-related questions."
"""
}

SYSTEM_PREFIXES = {
    'legal': '⚖️ **Legal Response:**\n\n',
    'medical': '🏥 **Medical Information:**\n\n',
    'disaster': '🌊 **Disaster Management:**\n\n',
    'all': '🌐 **Integrated Response:**\n\n'
}

# System states
system_states = {
    'legal': {'ready': True, 'loading': False, 'status_message': 'Ready', 'chunks_count': 0, 'pdf_files': []},
    'medical': {'ready': True, 'loading': False, 'status_message': 'Ready', 'chunks_count': 0, 'pdf_files': []},
    'disaster': {'ready': True, 'loading': False, 'status_message': 'Ready', 'chunks_count': 0, 'pdf_files': []},
    'all': {'ready': True, 'loading': False, 'status_message': 'Ready', 'chunks_count': 0, 'pdf_files': []}
}

# In-memory session storage (consider using Redis/DB for production)
chat_sessions = defaultdict(lambda: defaultdict(lambda: {
    'messages': [],
    'created_at': datetime.now(),
    'last_updated': datetime.now(),
    'title': 'New Chat',
    'message_count': 0
}))

def generate_chat_title(messages, system_key):
    """Generate a title for the chat based on first message"""
    if not messages:
        return f"New {SYSTEMS[system_key]['name']} Chat"
    
    first_message = next((m['content'] for m in messages if m['role'] == 'user'), None)
    if not first_message:
        return f"New {SYSTEMS[system_key]['name']} Chat"
    
    words = first_message.split()[:6]
    title = ' '.join(words)
    if len(title) > 40:
        title = title[:37] + "..."
    
    return f"{SYSTEMS[system_key]['icon']} {title.capitalize()}"

def sanitize_input(text):
    """Sanitize user input to prevent injection"""
    if not text:
        return ""
    # Remove any potentially dangerous characters
    import re
    text = re.sub(r'[<>]', '', text)
    return text.strip()

def stream_mistral(question, system_prompt, retry_count=0):
    """
    Streams tokens from Mistral API using SSE
    Yields chunks as: data: {"token": "..."}\n\n
    Ends with:        data: {"done": true}\n\n
    """
    if not MISTRAL_API_KEY:
        yield 'data: {"error": "Mistral API key not configured."}\n\n'
        return

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": True
    }

    max_retries = 3
    try:
        with requests.post(MISTRAL_API_URL, json=payload, headers=headers,
                          stream=True, timeout=90) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        yield 'data: {"done": true}\n\n'
                        return
                    
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk['choices'][0]['delta']
                        token = delta.get('content', '')
                        if token:
                            yield f'data: {json.dumps({"token": token})}\n\n'
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.warning(f"Error parsing chunk: {e}")
                        continue

    except requests.exceptions.Timeout:
        if retry_count < max_retries:
            logger.info(f"Timeout, retrying... (attempt {retry_count + 1})")
            time.sleep(2 ** retry_count)  # Exponential backoff
            yield from stream_mistral(question, system_prompt, retry_count + 1)
        else:
            yield 'data: {"error": "Request timed out. Please try again."}\n\n'
            
    except requests.exceptions.RequestException as e:
        if retry_count < max_retries:
            logger.info(f"Request failed, retrying... (attempt {retry_count + 1})")
            time.sleep(2 ** retry_count)
            yield from stream_mistral(question, system_prompt, retry_count + 1)
        else:
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
            
    except Exception as e:
        logger.error(f"Unexpected error in stream_mistral: {e}")
        yield f'data: {json.dumps({"error": "An unexpected error occurred"})}\n\n'

    yield 'data: {"done": true}\n\n'

# Routes
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', systems=SYSTEMS)

@app.route('/api/ask/stream', methods=['POST'])
@limiter.limit("30 per minute")
def ask_question_stream():
    """Streaming endpoint - returns SSE stream of tokens"""
    try:
        data = request.json
        system_key = data.get('system', 'legal')
        question = sanitize_input(data.get('question', ''))
        session_id = data.get('session_id')

        logger.info(f"Stream request - System: {system_key}, Session: {session_id}, Question: {question[:50]}...")

        if system_key not in SYSTEMS:
            return jsonify({'error': 'Invalid system'}), 400
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Store user message if session exists
        if session_id and session_id in chat_sessions[system_key]:
            session = chat_sessions[system_key][session_id]
            session['messages'].append({
                'role': 'user',
                'content': question,
                'timestamp': datetime.now().isoformat()
            })
            session['message_count'] += 1
            session['last_updated'] = datetime.now()
            session['title'] = generate_chat_title(session['messages'], system_key)

        system_prompt = SYSTEM_PROMPTS.get(system_key, SYSTEM_PROMPTS['all'])

        # Build question with conversation context
        final_question = question
        if session_id and session_id in chat_sessions[system_key]:
            recent = chat_sessions[system_key][session_id]['messages'][-6:]
            if len(recent) > 1:
                context = "\n".join([f"{m['role']}: {m['content']}" for m in recent[:-1]])
                final_question = f"Previous conversation:\n{context}\n\nUser: {question}"

        prefix = SYSTEM_PREFIXES.get(system_key, '')

        def generate():
            full_answer = prefix
            yield f'data: {json.dumps({"token": prefix})}\n\n'

            for chunk in stream_mistral(final_question, system_prompt):
                if chunk.startswith('data: '):
                    try:
                        parsed = json.loads(chunk[6:])
                        if 'token' in parsed:
                            full_answer += parsed['token']
                        elif parsed.get('done'):
                            # Save complete answer to session
                            if session_id and session_id in chat_sessions[system_key]:
                                chat_sessions[system_key][session_id]['messages'].append({
                                    'role': 'assistant',
                                    'content': full_answer,
                                    'timestamp': datetime.now().isoformat()
                                })
                                chat_sessions[system_key][session_id]['message_count'] += 1
                                chat_sessions[system_key][session_id]['last_updated'] = datetime.now()
                                logger.info(f"Saved response for session {session_id}")
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error parsing chunk: {e}")
                        pass
                yield chunk

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in ask_question_stream: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
@limiter.limit("30 per minute")
def ask_question():
    """Non-streaming fallback endpoint"""
    try:
        data = request.json
        system_key = data.get('system', 'legal')
        question = sanitize_input(data.get('question', ''))
        session_id = data.get('session_id')

        logger.info(f"Non-stream request - System: {system_key}, Session: {session_id}")

        if system_key not in SYSTEMS:
            return jsonify({'error': 'Invalid system'}), 400
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        if session_id and session_id in chat_sessions[system_key]:
            session = chat_sessions[system_key][session_id]
            session['messages'].append({
                'role': 'user', 
                'content': question, 
                'timestamp': datetime.now().isoformat()
            })
            session['message_count'] += 1
            session['last_updated'] = datetime.now()
            session['title'] = generate_chat_title(session['messages'], system_key)

        system_prompt = SYSTEM_PROMPTS.get(system_key, SYSTEM_PROMPTS['all'])

        final_question = question
        if session_id and session_id in chat_sessions[system_key]:
            recent = chat_sessions[system_key][session_id]['messages'][-5:]
            if len(recent) > 1:
                context = "\n".join([f"{m['role']}: {m['content']}" for m in recent[:-1]])
                final_question = f"Previous conversation:\n{context}\n\nUser: {question}"

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}", 
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": final_question}
            ],
            "temperature": 0.7, 
            "max_tokens": 1000
        }
        
        resp = requests.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        answer = resp.json()['choices'][0]['message']['content']
        answer = SYSTEM_PREFIXES.get(system_key, '') + answer

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
        logger.error(f"Error in ask_question: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<system_key>')
def get_system_status(system_key):
    """Get system status"""
    if system_key not in system_states:
        return jsonify({'error': 'Invalid system'}), 404
    
    state = system_states[system_key]
    return jsonify({
        'ready': state['ready'], 
        'loading': state['loading'], 
        'message': state['status_message'],
        'chunks_count': state.get('chunks_count', 0), 
        'pdf_files': state.get('pdf_files', []),
        'system_info': SYSTEMS[system_key]
    })

@app.route('/api/sessions/<system_key>', methods=['GET'])
def get_sessions(system_key):
    """Get all sessions for a system"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    sessions_list = []
    for session_id, sd in chat_sessions[system_key].items():
        preview = ""
        if sd['messages']:
            first_msg = sd['messages'][0]['content']
            preview = first_msg[:50] + '...' if len(first_msg) > 50 else first_msg
        
        sessions_list.append({
            'id': session_id, 
            'title': sd.get('title', f"New Chat"),
            'created_at': sd['created_at'].isoformat(), 
            'last_updated': sd['last_updated'].isoformat(),
            'message_count': len(sd['messages']), 
            'preview': preview or 'Empty chat'
        })
    
    sessions_list.sort(key=lambda x: x['last_updated'], reverse=True)
    return jsonify({'sessions': sessions_list})

@app.route('/api/sessions/<system_key>', methods=['POST'])
def create_session(system_key):
    """Create a new session"""
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
    
    logger.info(f"Created new session {session_id} for system {system_key}")
    return jsonify({'session_id': session_id, 'title': chat_sessions[system_key][session_id]['title']})

@app.route('/api/sessions/<system_key>/<session_id>', methods=['GET'])
def get_session(system_key, session_id):
    """Get a specific session"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    if session_id not in chat_sessions[system_key]:
        return jsonify({'error': 'Session not found'}), 404
    
    sd = chat_sessions[system_key][session_id]
    return jsonify({
        'id': session_id, 
        'title': sd.get('title', 'New Chat'), 
        'messages': sd['messages'],
        'created_at': sd['created_at'].isoformat(), 
        'last_updated': sd['last_updated'].isoformat(),
        'message_count': len(sd['messages'])
    })

@app.route('/api/sessions/<system_key>/<session_id>', methods=['DELETE'])
def delete_session(system_key, session_id):
    """Delete a session"""
    if system_key not in SYSTEMS:
        return jsonify({'error': 'Invalid system'}), 404
    
    if session_id in chat_sessions[system_key]:
        del chat_sessions[system_key][session_id]
        logger.info(f"Deleted session {session_id} from system {system_key}")
        return jsonify({'success': True})
    
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/systems', methods=['GET'])
def get_systems():
    """Get all systems information"""
    return jsonify({
        k: {
            'info': SYSTEMS[k], 
            'status': {
                'ready': system_states[k]['ready'],
                'loading': system_states[k]['loading'], 
                'chunks_count': system_states[k].get('chunks_count', 0)
            }
        } for k in SYSTEMS
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'mistral_api': 'configured' if MISTRAL_API_KEY else 'missing',
        'systems': len(SYSTEMS),
        'timestamp': datetime.now().isoformat()
    }), 200

# Session cleanup (optional)
def cleanup_old_sessions(days=30):
    """Clean up sessions older than specified days"""
    try:
        cutoff = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for system in SYSTEMS:
            sessions_to_delete = []
            for session_id, session in chat_sessions[system].items():
                if session['last_updated'] < cutoff:
                    sessions_to_delete.append(session_id)
            
            for session_id in sessions_to_delete:
                del chat_sessions[system][session_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old sessions")
            
    except Exception as e:
        logger.error(f"Error in session cleanup: {e}")

# Create necessary directories
def setup_directories():
    """Create necessary directories if they don't exist"""
    for system in ['legal', 'medical', 'disaster']:
        os.makedirs(SYSTEMS[system]['pdf_folder'], exist_ok=True)
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

if __name__ == "__main__":
    # Setup
    setup_directories()
    
    # Print startup information
    print("\n" + "="*50)
    print("🚀 TRINETRA AI SYSTEM STARTING")
    print("="*50)
    print(f"✅ Mistral API configured (Model: {MISTRAL_MODEL})")
    print(f"✅ Streaming: ON")
    print(f"✅ Systems loaded: {len(SYSTEMS)}")
    print(f"   - Legal System")
    print(f"   - Medical System")
    print(f"   - Disaster Management")
    print(f"   - All-in-One System")
    print("="*50)
    print("\n🌐 Server running at: http://localhost:5000")
    print("📝 API endpoints available at: /api/*")
    print("="*50 + "\n")
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

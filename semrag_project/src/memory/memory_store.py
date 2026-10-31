"""Long-term memory in SQLite: facts the user asks the assistant to remember, plus saved chats.

Short-term memory is the recent chat history that app.py puts into each prompt;
everything here survives restarts.
"""
import re
import sqlite3
import threading
from contextlib import closing
from datetime import datetime

_db_path = None
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    system TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
"""

def init(db_path):
    global _db_path
    _db_path = db_path
    with _lock, closing(sqlite3.connect(_db_path)) as conn:
        conn.executescript(SCHEMA)
        conn.commit()

def _execute(sql, params=()):
    with _lock, closing(sqlite3.connect(_db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
        return rows, cur.lastrowid, cur.rowcount

# ---------------------------------------------------------------- facts

def add_fact(fact):
    existing = [f for f in list_facts() if f['fact'].lower() == fact.lower()]
    if existing:
        return existing[0]['id']
    _, fact_id, _ = _execute("INSERT INTO facts (fact, created_at) VALUES (?, ?)",
                             (fact, datetime.now().isoformat()))
    return fact_id

def list_facts():
    rows, _, _ = _execute("SELECT id, fact, created_at FROM facts ORDER BY id")
    return rows

def delete_fact(fact_id):
    _, _, count = _execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    return count

def forget_matching(text):
    _, _, count = _execute("DELETE FROM facts WHERE fact LIKE ?", (f"%{text}%",))
    return count

def clear_facts():
    _, _, count = _execute("DELETE FROM facts")
    return count

# ---------------------------------------------------------------- chat sessions

def save_session(system, session_id, title, created_at, last_updated):
    _execute("""INSERT INTO chat_sessions (id, system, title, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET title = excluded.title, last_updated = excluded.last_updated""",
             (session_id, system, title, created_at.isoformat(), last_updated.isoformat()))

def add_message(session_id, role, content, timestamp):
    _execute("INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
             (session_id, role, content, timestamp))

def delete_session(session_id):
    _execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    _execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

def load_sessions():
    """Returns {system: {session_id: {'title', 'messages', 'created_at', 'last_updated'}}}"""
    sessions, _, _ = _execute("SELECT * FROM chat_sessions")
    messages, _, _ = _execute("SELECT session_id, role, content, timestamp FROM chat_messages ORDER BY id")

    result = {}
    for s in sessions:
        result.setdefault(s['system'], {})[s['id']] = {
            'title': s['title'],
            'messages': [],
            'created_at': datetime.fromisoformat(s['created_at']),
            'last_updated': datetime.fromisoformat(s['last_updated'])
        }
    by_id = {sid: data for system in result.values() for sid, data in system.items()}
    for m in messages:
        if m['session_id'] in by_id:
            by_id[m['session_id']]['messages'].append(
                {'role': m['role'], 'content': m['content'], 'timestamp': m['timestamp']})
    return result

# ---------------------------------------------------------------- chat commands

LIST_RE = re.compile(r"^\s*(what (do|did) you remember|what('s| is) in your memory|show (me )?(your )?memory)\b", re.I)
RECALL_QUESTION_RE = re.compile(r"^\s*(do|did|can|could|will|would)\s+you\s+remember\b", re.I)
REMEMBER_RE = re.compile(r"\b(?:remember|keep in mind)\b(?:\s+that)?\s*[:,\-]?\s*(.+)", re.I)
FORGET_RE = re.compile(r"^\s*(?:please\s+)?forget\b(?:\s+(?:that|about))?\s*(.*)", re.I)

def handle_command(question):
    """Handle remember / forget / list commands. Returns a reply, or None if the question is not a command."""
    if LIST_RE.search(question):
        facts = list_facts()
        if not facts:
            return "I don't have anything in long-term memory yet. Say \"remember ...\" to teach me something."
        return "Here is what I remember:\n" + "\n".join(f"• {f['fact']}" for f in facts)

    forget = FORGET_RE.match(question)
    if forget:
        target = forget.group(1).strip(" .!?\"'")
        if target.lower() in ("", "everything", "all", "it all", "all memories", "your memory"):
            clear_facts()
            return "Done. My long-term memory is now empty."
        count = forget_matching(target)
        if count:
            return f"Okay, I forgot {count} memory item(s) about \"{target}\"."
        return f"I didn't find anything about \"{target}\" in my memory."

    remember = REMEMBER_RE.search(question)
    if remember and not RECALL_QUESTION_RE.match(question) and not question.rstrip().endswith("?"):
        fact = remember.group(1).strip(" .!\"'")
        if fact:
            add_fact(fact)
            return f"Got it, I'll remember that: \"{fact}\"."
    return None

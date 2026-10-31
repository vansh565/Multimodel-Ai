from src.llm.llm_client import generate, generate_stream

def format_memory(facts):
    if not facts:
        return ""
    lines = "\n".join(f"- {fact}" for fact in facts)
    return f"""Long-term memory (facts the user told you to remember; "you"/"u" in them means you, the assistant.
These facts are true and override anything else you believe, but only mention them when the question is about them):
{lines}
"""

def build_prompt(query, local_data, global_data, instructions="", history="", memory=()):
    context = "\n".join(local_data) + "\n" + "\n".join(global_data)

    return f"""
{instructions}
{format_memory(memory)}
You MUST answer in ONLY 3 lines.
Do NOT explain.
Keep answer short.
donot use page number ,use only the context of the pdf, the long-term memory and the conversation
Context:
{context}

{history}
Question:
{query}

Answer:
"""

def generate_answer(query, local_data, global_data, instructions="", history="", memory=()):
    response = generate(build_prompt(query, local_data, global_data, instructions, history, memory))

    return " ".join(response.split()[:25])

def generate_answer_stream(query, local_data, global_data, instructions="", history="", memory=()):
    yield from generate_stream(build_prompt(query, local_data, global_data, instructions, history, memory))

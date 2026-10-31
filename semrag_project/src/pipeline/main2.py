import requests
import speech_recognition as sr
import pyttsx3

# 🎤 Voice
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 180)

def speak(text):
    print("Bot:", text)
    engine.stop()
    engine.say(text)
    engine.runAndWait()


def get_input():
    mode = input("(v) voice / (t) text: ").strip()

    if mode == "t":
        return input("You: ").lower()

    elif mode == "v":
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
                text = recognizer.recognize_google(audio)
                print("You:", text)
                return text.lower()
            except:
                return ""

    return ""


# 📄 LOAD DATASET
with open("data/medical.txt", "r", encoding="utf-8") as f:
    knowledge = f.read()


# 🤖 LLM (Mistral)
def llm(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 80
            }
        }
    )
    return res.json()["response"]


# 🧠 QUESTION GENERATOR
asked = set()

def ask_question(problem, history):
    prompt = f"""
You are a medical assistant.

Problem: {problem}
Conversation: {history}

Ask ONE short symptom-related question.
Do NOT repeat.
"""

    q = llm(prompt).strip().split("?")[0] + "?"

    if q in asked:
        return "Any other symptom?"

    asked.add(q)
    return q


# 🧠 FINAL DOCTOR DECISION (LLM + DATASET)
def get_doctor(history):
    prompt = f"""
You are a medical assistant.

DATASET:
{knowledge}

PATIENT INFO:
{history}

RULES:
- Choose doctor ONLY from dataset
- Do NOT invent names
- Match symptoms carefully

OUTPUT:
Doctor: ___
Profession: ___
"""

    return llm(prompt)


# 🔁 MAIN LOOP
while True:
    history = []
    asked.clear()

    query = get_input()

    if not query:
        continue

    if "stop" in query:
        break

    history.append(query)

    # 🔥 ASK ONLY 2 QUESTIONS
    for _ in range(2):
        q = ask_question(query, history)
        speak(q)

        ans = get_input()

        if not ans:
            ans = ""

        history.append(ans)

    # 🔥 FINAL RESULT (LLM decides using dataset)
    result = get_doctor(history)
    speak(result)

    print("\n--- Session Complete ---\n")
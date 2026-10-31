# src/llm/llm_client.py
import os
import requests
import json

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def generate(prompt, model=OLLAMA_MODEL):
    """Generate response using Ollama with Mistral model"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500
                }
            },
            timeout=180  # first call can be slow while Ollama loads the model
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        elif response.status_code == 404:
            return f"Error: Ollama model '{model}' not found. Run 'ollama pull {model}'."
        else:
            print(f"Ollama API error: {response.status_code}")
            return "Error: Unable to connect to Ollama. Please make sure Ollama is running."
    
    except requests.exceptions.ConnectionError:
        print("Connection error: Make sure Ollama is running with 'ollama serve'")
        return "Error: Cannot connect to Ollama. Please run 'ollama serve' in terminal."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Error: {str(e)}"

def generate_stream(prompt, model=OLLAMA_MODEL):
    """Generate streaming response using Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500
                }
            },
            stream=True,
            timeout=180
        )

        if response.status_code == 404:
            yield f"Error: Ollama model '{model}' not found. Run 'ollama pull {model}'."
            return
        if response.status_code != 200:
            yield f"Error: Ollama API error {response.status_code}"
            return

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    yield data.get("response", "")
                except:
                    pass
    except requests.exceptions.ConnectionError:
        print("Connection error: Make sure Ollama is running with 'ollama serve'")
        yield "Error: Cannot connect to Ollama. Please run 'ollama serve' in terminal."
    except Exception as e:
        print(f"Streaming error: {e}")
        yield f"Error: {str(e)}"

def check_mistral_available(model_name=OLLAMA_MODEL):
    """Check if Mistral model is available in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if model_name.lower() in model.get("name", "").lower():
                    return True
        return False
    except:
        return False
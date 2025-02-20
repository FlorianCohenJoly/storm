import requests
import json
import litellm

MODELE_OLLAMA = "llama3.2:latest"  # Assurez-vous que ce modèle existe dans Ollama
PORT_OLLAMA = 11434

def generer_texte(prompt, model=MODELE_OLLAMA, port=PORT_OLLAMA, max_tokens=500, temperature=0.7):
    url = f"http://localhost:{port}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,  # Important: False pour LiteLLM
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)

        result = response.json()
        # print(f"Réponse brute d'Ollama : {result}")  # Décommentez pour le débogage

        # Gestion plus robuste de la réponse JSON
        if "response" in result:
            return result["response"]
        elif "error" in result:
            print(f"Erreur d'Ollama: {result['error']}")
            return None  # Ou raise une exception plus spécifique
        else:
            print("Réponse JSON inattendue d'Ollama:", result)
            return None  # Ou raise une exception

    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête à Ollama: {e}")
        return None

class OllamaWrapper:
    def __init__(self, model=MODELE_OLLAMA, temperature=0.7, max_tokens=500):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = {"temperature": temperature, "max_tokens": max_tokens}

    def __call__(self, prompt, **kwargs):
        self.kwargs.update(kwargs)  # Fusionne les kwargs passés à l'appel avec les kwargs initiaux
        max_tokens = self.kwargs.get("max_tokens", self.max_tokens)
        temperature = self.kwargs.get("temperature", self.temperature)

        return generer_texte(prompt, model=self.model, port=PORT_OLLAMA, max_tokens=max_tokens, temperature=temperature)

# Intégration avec LiteLLM
litellm.custom_llm_provider = OllamaWrapper()  # Utilisation de la classe wrapper

try:
    response = litellm.completion(model=MODELE_OLLAMA, prompt="Votre prompt ici", temperature=0.8, max_tokens=600)  # Utilisation de LiteLLM
    print(response)
except Exception as e:
    print(f"Erreur avec LiteLLM: {e}")


# Test direct de la classe wrapper (pour comparaison)
# ollama_instance = OllamaWrapper()
# reponse_directe = ollama_instance("Un autre prompt ici", temperature=0.9, max_tokens=700)
# print("Réponse directe:", reponse_directe)
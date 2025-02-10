import requests

MODELE_OLLAMA = "llama3.2:latest"  # Remplacez par le nom de votre modèle
PORT_OLLAMA = 11434  # Remplacez si nécessaire

def generer_texte(prompt, model=MODELE_OLLAMA, port=PORT_OLLAMA):
    url = f"http://localhost:{port}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        texte_genere = result["response"]
        return texte_genere
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête : {e}")
        return None
    except (KeyError, TypeError) as e:
        print(f"Erreur lors du traitement de la réponse : {e}")
        return None
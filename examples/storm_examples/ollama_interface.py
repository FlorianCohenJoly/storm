import requests
import json  # Importez le module json

MODELE_OLLAMA = "llama3.2:latest"
PORT_OLLAMA = 11434

def generer_texte(prompt, model=MODELE_OLLAMA, port=PORT_OLLAMA):
    url = f"http://localhost:{port}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "format": "json",  # Important : demander le format JSON
        "stream": False,
        "temperature": 0.7,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Vérifie le code de statut HTTP

        try:  # Bloc try pour le parsing JSON
            result = response.json()
            print(f"Réponse brute d'Ollama : {result}")  # Afficher pour le débogage

            try: # Bloc try pour la clé 'response'
                texte_genere = result["response"]
                return texte_genere
            except KeyError as e:
                print(f"Erreur: Clé 'response' non trouvée dans la réponse JSON: {e}")
                return None

        except json.JSONDecodeError as e:
            print(f"Erreur: Réponse d'Ollama non parsable en JSON: {e}. Contenu de la réponse: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête à Ollama: {e}")
        return None
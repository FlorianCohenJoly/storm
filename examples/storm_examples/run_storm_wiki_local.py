import streamlit as st
import json
import re  # Importation du module re pour les expressions régulières
from my_engine import MyEngine  # Assurez-vous que ce module est correctement importé
from knowledge_storm import STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from ollama_interface import generer_texte  # Assurez-vous que ce module est correctement importé


def run_app():
    st.title("Mon Application avec Ollama et STORM")

    topic = st.text_input("Entrez un sujet :")

    # Configuration des modèles de langage (à l'extérieur de la condition if topic)
    lm_configs = STORMWikiLMConfigs()
    modele_ollama = LitellmModel(model="llama3.2:latest", max_tokens=5000)  # Remplacez par votre modèle si nécessaire

    lm_configs.set_conv_simulator_lm(modele_ollama)
    lm_configs.set_question_asker_lm(modele_ollama)
    lm_configs.set_outline_gen_lm(modele_ollama)
    lm_configs.set_article_gen_lm(modele_ollama)
    lm_configs.set_article_polish_lm(modele_ollama)

    if topic:
        engine = MyEngine(lm_configs)  # Initialisation de MyEngine à l'intérieur du if topic
        prompt_json = f"""
        Répondez à la question suivante au format JSON. Développez chaque section avec des détails pertinents, des exemples concrets et des sources crédibles.

        {{
          "title": "Le titre de l'article",
          "description": "Une description détaillée de l'article",
          "categories": ["Catégorie 1", "Catégorie 2", ...],
          "subtopics": [
            {{"topic": "Sous-sujet 1", "description": "Description du sous-sujet 1"}},
            {{"topic": "Sous-sujet 2", "description": "Description du sous-sujet 2"}},
            ...
          ],
          "key-terms": [
            {{"term": "Terme clé 1", "definition": "Définition du terme clé 1"}},
            {{"term": "Terme clé 2", "definition": "Définition du terme clé 2"}},
            ...
          ],
          "useful-links": [
            {{"link": "Lien 1", "description": "Description du lien 1"}},
            {{"link": "Lien 2", "description": "Description du lien 2"}},
            ...
          ]
        }}

        Question: {topic}
        """

        with st.spinner("Génération du texte en cours..."):
            result_text = generer_texte(prompt_json)  # Récupérer le texte brut

            if result_text:
                try:
                    # Extraction du JSON avec une expression régulière (plus robuste)
                    match = re.search(r"\{.*\}", result_text, re.DOTALL)  # Recherche du JSON entre accolades
                    if match:
                        json_str = match.group(0)
                        try: # Ajout d'un deuxième try pour gérer les erreurs JSONDecodeError plus spécifiques.
                            result = json.loads(json_str)  # Chargement du JSON
                            st.write(result)

                            # Enregistrement du JSON
                            try:
                                with open("reponse.json", "w", encoding="utf-8") as f:
                                    json.dump(result, f, ensure_ascii=False, indent=4)
                                st.success("Réponse enregistrée dans reponse.json")
                            except Exception as e:
                                st.error(f"Erreur lors de l'enregistrement du fichier JSON : {e}")
                        except json.JSONDecodeError as e:
                            st.error(f"Erreur de décodage JSON : {e}")
                            st.write("JSON extrait (avant correction) :", json_str) # Afficher le JSON extrait avant correction
                            # Tentative de correction du JSON (suppression des virgules et crochets en trop)
                            json_str = json_str.replace(",]", "]").replace(",}", "}") # Correction des virgules et accolades
                            try:
                                result = json.loads(json_str)
                                st.write(result)
                                with open("reponse.json", "w", encoding="utf-8") as f:
                                    json.dump(result, f, ensure_ascii=False, indent=4)
                                st.success("Réponse enregistrée dans reponse.json (après correction)")
                            except json.JSONDecodeError as e:
                                st.error(f"Erreur de décodage JSON (après correction) : {e}")
                                st.write("JSON extrait (après correction) :", json_str) # Afficher le JSON extrait après correction
                    else:
                        st.error("Aucun JSON valide trouvé dans la réponse.")
                        st.write("Réponse brute :", result_text)  # Afficher la réponse brute

                except json.JSONDecodeError as e:
                    st.error(f"Erreur de décodage JSON : {e}")
                    st.write("Réponse brute :", result_text)  # Afficher la réponse brute
            else:
                st.error("Erreur lors de la génération du texte.")


if __name__ == "__main__":
    run_app()
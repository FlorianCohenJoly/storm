import streamlit as st
from knowledge_storm import STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from my_engine import MyEngine, EcrivainAgent, ExpertAgent
import json

def run_app():
    st.title("Mon Application avec Ollama et STORM")
    topic = st.text_input("Entrez un sujet :")

    lm_configs = STORMWikiLMConfigs()
    modele_ollama = LitellmModel(model="llama3.2:latest", max_tokens=5000)

    lm_configs.set_conv_simulator_lm(modele_ollama)
    lm_configs.set_question_asker_lm(modele_ollama)
    lm_configs.set_outline_gen_lm(modele_ollama)
    lm_configs.set_article_gen_lm(modele_ollama)
    lm_configs.set_article_polish_lm(modele_ollama)

    conversation_history = []

    if topic:
        ecrivain = EcrivainAgent(topic, modele_ollama)
        expert = ExpertAgent(topic, modele_ollama)
        engine = MyEngine(lm_configs, ecrivain, expert)

        conversation_history = engine.run(topic)
        plan = engine.run_outline_generation_module()
        article = engine.run_article_generation_module(plan=plan)
        final_article = engine.run_article_polishing_module(article=article)

        if conversation_history:
            st.write("Historique de la conversation :")
            for turn in conversation_history:
                st.write(f"{turn.role} : {turn.utterance}")
        else:
            st.write("Aucune conversation n'a eu lieu.")

        if plan:
            st.write("Plan :")
            st.write(plan)
        else:
            st.write("Le modèle n'a pas généré de plan.")

        

        if article:
            st.write("### Article :")
            try:
                article_data = json.loads(article)  # Convertir le JSON en dictionnaire
                st.json(article_data)  # Afficher sous format JSON structuré
            except json.JSONDecodeError:
                st.markdown(article)  # Si c'est du texte brut, l'afficher proprement
        else:
            st.write("Le modèle n'a pas généré d'article.")

        if final_article:
            st.write("### Article final :")
            try:
                final_article_data = json.loads(final_article)
                st.json(final_article_data)
            except json.JSONDecodeError:
                st.markdown(final_article)
        else:
            st.write("Le modèle n'a pas généré d'article final.")

if __name__ == "__main__":
    run_app()
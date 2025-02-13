import streamlit as st
from knowledge_storm import STORMWikiLMConfigs
from knowledge_storm.dataclass import KnowledgeBase
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
        knowledge_base = KnowledgeBase(topic, modele_ollama, 0, None)  # Définir knowledge_base ICI
        knowledge_base.knowledge = {}  # Correction : Ajout manuel de l'attribut missing
        ecrivain = EcrivainAgent(topic, modele_ollama)
        expert = ExpertAgent(topic, modele_ollama)
        engine = MyEngine(lm_configs, knowledge_base, ecrivain, expert)  # knowledge_base est maintenant définie

        conversation_history = engine.run(topic, personas=["persona1", "persona2"])  # Ajout de personas

        if conversation_history:
            st.write("Historique de la conversation :")
            for turn in conversation_history:
                print(turn)  # Débogage
                st.write(f"{turn.role}: {turn.utterance}")
        else:
            st.write("Aucune conversation n'a eu lieu.")

        plan = engine.run_outline_generation_module()
        if plan:
            st.write("Plan :")
            st.write(plan.utterance)  # Accéder à l'attribut utterance de l'objet ConversationTurn
        else:
            st.write("Le modèle n'a pas généré de plan.")

        article = engine.run_article_generation_module(plan=plan)
        if article:
            st.write("### Article :")
            try:
                article_data = json.loads(article.utterance)  # Accéder à l'attribut utterance et parser le JSON
                st.json(article_data)
            except json.JSONDecodeError:
                st.markdown(article.utterance)
        else:
            st.write("Le modèle n'a pas généré d'article.")

        final_article = engine.run_article_polishing_module(article=article)
        if final_article:
            st.write("### Article final :")
            try:
                final_article_data = json.loads(final_article.utterance)  # Accéder à l'attribut utterance et parser le JSON
                st.json(final_article_data)
            except json.JSONDecodeError:
                st.markdown(final_article.utterance)
        else:
            st.write("Le modèle n'a pas généré d'article final.")

if __name__ == "__main__":
    run_app()
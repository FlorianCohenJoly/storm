import streamlit as st
from knowledge_storm import STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from my_engine import MyEngine, EcrivainAgent, ExpertAgent

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

    if topic:
        ecrivain = EcrivainAgent(topic, modele_ollama)
        expert = ExpertAgent(topic, modele_ollama)
        engine = MyEngine(lm_configs, ecrivain, expert)

        conversation_history = engine.run(topic)
        plan = engine.run_outline_generation_module()
        article = engine.run_article_generation_module(plan=plan)
        final_article = engine.run_article_polishing_module(article=article)

        st.write("Historique de la conversation :", conversation_history)
        st.write("Plan :", plan)
        st.write("Article :", article)
        st.write("Article final :", final_article)

if __name__ == "__main__":
    run_app()
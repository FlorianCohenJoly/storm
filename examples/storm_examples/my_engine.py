from knowledge_storm.storm_wiki.engine import Engine
from knowledge_storm.dataclass import Information, KnowledgeBase, ConversationTurn
from ollama_interface import generer_texte
from dataclasses import dataclass, field
from typing import Tuple, Dict
import streamlit as st
from abc import ABC, abstractmethod
import json

@dataclass(frozen=True)
class Information:
    content: str
    citation_uuid: int
    meta: Tuple[Tuple[str, str], ...]

class Agent(ABC):
    def __init__(self, topic, role_name, action_description):
        self.topic = topic
        self.role_name = role_name
        self.action_description = action_description

    @abstractmethod
    def generate_utterance(self, input_data=None):
        pass

class EcrivainAgent(Agent):
    def __init__(self, topic, knowledge_base_lm):
        super().__init__(topic, "Ecrivain Wikipedia", "Rédige l'article")
        self.knowledge_base = KnowledgeBase(topic, knowledge_base_lm, 0, None)
        self.conversation_history = []

    def generer_questions(self):
        return [
            "Quels sont les principaux aspects historiques de ce sujet ?",
            "Quels sont les impacts économiques de ce sujet ?",
            "Quels sont les aspects culturels de ce sujet ?",
            "Quelles sont les controverses ou débats liés à ce sujet ?"
        ]

    def generer_plan(self):
        knowledge = "\n".join([info.content for info in self.knowledge_base.knowledge.values()])
        plan = generer_texte(f"Génère un plan pour un article sur {self.topic} basé sur les informations suivantes : {knowledge}")
        return plan

    def generer_article(self, plan):
        knowledge = "\n".join([info.content for info in self.knowledge_base.knowledge.values()])
        article = generer_texte(f"Rédige un article détaillé et bien structuré sur {self.topic}, basé sur le plan suivant : {plan} et les informations suivantes : {knowledge}")
        return article

    def generate_utterance(self, input_data=None):
        if input_data is None or not input_data:
            return self.generer_questions()[0]
        elif isinstance(input_data, dict) and "type" in input_data:
            if input_data["type"] == "plan":
                return self.generer_plan()
            elif input_data["type"] == "article":
                return self.generer_article(input_data["plan"])
        return None

class ExpertAgent(Agent):
    def __init__(self, topic, knowledge_base_lm):
        super().__init__(topic, "Expert du sujet", "Fournit des informations")
        self.knowledge_base = KnowledgeBase(topic, knowledge_base_lm, 0, None)
        self.conversation_history = []

    def repondre_question(self, question):
        prompt = f"""
        Réponds à la question suivante au format JSON.  Utilise la structure suivante :

        ```json
        {{
          "question": "{question}",
          "answer": "Texte de la réponse"
        }}
        ```
        """
        reponse_json_str = generer_texte(prompt)

        try:
            reponse_json = json.loads(reponse_json_str)
            reponse = reponse_json["answer"]

            conversation_turn = ConversationTurn(
                role=self.role_name,
                utterance=reponse,
                raw_utterance=reponse_json_str,
                utterance_type="reponse"
            )
            self.conversation_history.append(conversation_turn)

            meta_tuple = tuple({ "question": question}.items())
            information = Information(
                content=reponse,
                citation_uuid=-1,
                meta=meta_tuple
            )

            try:
                path = self.topic
                self.knowledge_base.insert_information(path=path, information=information)
            except Exception as e:
                st.error(f"Erreur lors de l'insertion de l'information : {e}")
                st.exception(e)

            return conversation_turn
        except json.JSONDecodeError as e:
            print(f"Erreur de parsing JSON: {e}")
            print(f"Réponse brute du LLM: {reponse_json_str}")
            reponse = f"Erreur: Réponse du LLM non parsable en JSON.  Veuillez reformuler votre question."

            conversation_turn = ConversationTurn(
                role=self.role_name,
                utterance=reponse,
                raw_utterance=reponse_json_str,
                utterance_type="reponse"
            )
            return conversation_turn

    def generate_utterance(self, input_data=None):
        if isinstance(input_data, str):
            return self.repondre_question(input_data)
        return None

class KnowledgeBase:
    def __init__(self, topic, knowledge_base_lm, depth, parent):
        self.topic = topic
        self.knowledge_base_lm = knowledge_base_lm
        self.depth = depth
        self.parent = parent
        self.knowledge: Dict[str, Information] = {}

    def insert_information(self, path, information):
        self.knowledge[path] = information

class MyEngine(Engine):
    def __init__(self, lm_configs, ecrivain_agent, expert_agent):
        super().__init__(lm_configs)
        self.topic = None
        self.ecrivain_agent = ecrivain_agent
        self.expert_agent = expert_agent

    def run(self, topic, **kwargs):
        self.topic = topic
        self.ecrivain_agent.topic = topic
        self.expert_agent.topic = topic
        return self.run_knowledge_curation_module(**kwargs)

    def run_knowledge_curation_module(self, **kwargs):
        conversation_history = []
        questions = self.ecrivain_agent.generer_questions()
        for question in questions:
            reponse = self.expert_agent.generate_utterance(question)
            if reponse:
                conversation_history.append(reponse)
        return conversation_history

    def run_outline_generation_module(self, **kwargs):
        plan = self.ecrivain_agent.generate_utterance({"type": "plan"})
        return plan

    def run_article_generation_module(self, **kwargs):
        plan = kwargs.get("plan")
        article_parts = {
            "introduction": generer_texte(f"Rédige l'introduction de l'article sur {self.topic}, basé sur le plan suivant : {plan}"),
            "body_paragraphs": generer_texte(f"Rédige le corps de l'article sur {self.topic}, basé sur le plan suivant : {plan}"),
            "conclusion": generer_texte(f"Rédige la conclusion de l'article sur {self.topic}, basé sur le plan suivant : {plan}")
        }
        article = f"{article_parts['introduction']}\n\n{article_parts['body_paragraphs']}\n\n{article_parts['conclusion']}"
        return article

    def run_article_polishing_module(self, **kwargs):
        article = kwargs.get("article")
        return article
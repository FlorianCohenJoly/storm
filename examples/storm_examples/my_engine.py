from knowledge_storm.storm_wiki.engine import Engine
from knowledge_storm.dataclass import Information, KnowledgeBase, ConversationTurn
from ollama_interface import generer_texte
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
from abc import ABC, abstractmethod
import json

@dataclass
class ConversationTurn:
    role: str  # Ajout de l'attribut role
    utterance: str
    raw_utterance: str = ""  # Valeur par défaut pour raw_utterance
    utterance_type: str = ""  # Valeur par défaut pour utterance_type

@dataclass(frozen=True)
class Information:
    content: str
    citation_uuid: int
    meta: Tuple[Tuple[str, str], ...]

class Agent(ABC):
    def __init__(self, topic, engine, action_description, personas=None):
        self.topic = topic
        self.engine = engine
        self.action_description = action_description
        self.knowledge_base = None
        self.personas = personas if personas is not None else []
        self.role_name = self.__class__.__name__ # Le nom de la classe comme role


    @abstractmethod
    def run(self, *args, **kwargs):
        pass

class EcrivainAgent(Agent):
    def __init__(self, topic, engine, personas=None):
        super().__init__(topic, engine, "Écrit des articles Wikipedia", personas)

    def run(self, input_data=None):
        return self.generate_utterance(input_data)

    def generer_questions(self):
        questions = []
        for persona in self.personas:
            questions.extend([
                f"Quels sont les principaux aspects historiques de ce sujet du point de vue de {persona}?",
                f"Quels sont les impacts économiques de ce sujet selon {persona}?",
                f"Quels sont les aspects culturels de ce sujet pour {persona}?",
                f"Quelles sont les controverses ou débats liés à ce sujet selon {persona}?"
            ])
        return questions

    def generer_plan(self):
        knowledge = "\n".join([info.content for info in self.knowledge_base.knowledge.values()])
        prompt = f"Génère un plan pour un article sur {self.topic} en prenant en compte les perspectives suivantes : {', '.join(self.personas)}.\nInformations : {knowledge}"
        return generer_texte(prompt)

    def generer_article(self, plan):
        knowledge = "\n".join([info.content for info in self.knowledge_base.knowledge.values()])
        prompt = f"Rédige un article détaillé sur {self.topic} en utilisant le plan suivant : {plan}.\nPerspectives : {', '.join(self.personas)}\nInformations : {knowledge}"
        return generer_texte(prompt)

    
    def generate_utterance(self, input_data=None):
        if input_data is None:
            return self.generer_questions()[0] if self.personas else "Aucune question générée."
        elif isinstance(input_data, dict) and "type" in input_data:
            if input_data["type"] == "plan":
                return self.generer_plan()
            elif input_data["type"] == "article":
                return self.generer_article(input_data["plan"])
        return None

class ExpertAgent(Agent):
    def __init__(self, topic, engine, personas=None):
        super().__init__(topic, engine, "Répond aux questions sur des sujets d'expertise", personas)

    def run(self, input_data=None):
        return self.generate_utterance(input_data)

    def repondre_question(self, question):
        prompt = f"Réponds à la question : {question}. Considère les perspectives des personas suivants : {', '.join(self.personas)}."
        return generer_texte(prompt)

    def generate_utterance(self, input_data=None):
        utterance = super().generate_utterance(input_data) # On récupère le texte généré
        if utterance:
            return ConversationTurn(role=self.role_name, utterance=utterance) # On retourne un conversation turn
        return None

class MyEngine(Engine):
    def __init__(self, lm_configs, knowledge_base, ecrivain_agent, expert_agent):
        super().__init__(lm_configs)
        self.knowledge_base = knowledge_base
        self.ecrivain_agent = ecrivain_agent
        self.expert_agent = expert_agent

    def run(self, topic, personas=[]):
        self.ecrivain_agent.topic = topic
        self.expert_agent.topic = topic
        self.ecrivain_agent.knowledge_base = self.knowledge_base
        self.expert_agent.knowledge_base = self.knowledge_base
        self.ecrivain_agent.personas = personas
        self.expert_agent.personas = personas
        return self.run_knowledge_curation_module()

    def run_knowledge_curation_module(self):
        questions = self.ecrivain_agent.generer_questions()
        conversation_history = []
        for q in questions:
            expert_response = self.expert_agent.generate_utterance(q)
            if expert_response: # Vérifier si la réponse existe
                conversation_history.append(expert_response) # Ajouter la réponse à l'historique
        return conversation_history

    def run_outline_generation_module(self):
        return self.ecrivain_agent.generate_utterance({"type": "plan"})

    def run_article_generation_module(self, plan):
        return self.ecrivain_agent.generate_utterance({"type": "article", "plan": plan})

    def run_article_polishing_module(self, article):
        return article
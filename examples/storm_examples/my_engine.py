from examples.storm_examples.ollama_interface import generer_texte
from knowledge_storm.storm_wiki.engine import Engine
from knowledge_storm.dataclass import KnowledgeBase, ConversationTurn
from knowledge_storm.storm_wiki.modules.knowledge_curation import ConvSimulator
from knowledge_storm.storm_wiki.modules.outline_generation import StormOutlineGenerationModule
from knowledge_storm.storm_wiki.modules.article_generation import StormArticleGenerationModule
from knowledge_storm.storm_wiki.modules.article_polish import StormArticlePolishingModule
import json

class MyEngine(Engine):
    def __init__(self, knowledge_base, conv_simulator, outline_generator, article_generator, article_polisher):
        super().__init__(None)  # On ne passe plus lm_configs
        self.knowledge_base = knowledge_base
        self.conv_simulator = conv_simulator
        self.outline_generator = outline_generator
        self.article_generator = article_generator
        self.article_polisher = article_polisher
        self.llm = generer_texte  # On utilise directement Ollama

    
    def run_knowledge_curation_module(self, personas=[]):
        try:
            conversation_history = []
            for _ in range(self.conv_simulator.max_turn):
                question = f"Quels sont les points clés à aborder sur {self.knowledge_base.topic}?"
                expert_response = self.llm(question)

                
                if expert_response:
                    conversation_history.append(ConversationTurn(role="Expert", utterance=expert_response))
                    reaction_prompt = f"Réagis à cette réponse en tant qu'écrivain Wikipédia : {expert_response}"
                    writer_response = self.conv_simulator.question_asker.run(reaction_prompt)
                    conversation_history.append(ConversationTurn(role="Écrivain", utterance=writer_response))
            
            return conversation_history
        except Exception as e:
            print(f"Erreur dans run_knowledge_curation_module: {e}")
            return []
    
    def run_outline_generation_module(self):
        return self.outline_generator.generate_outline(self.knowledge_base.topic, self.knowledge_base)
    
    def run_article_generation_module(self, plan):
        return self.article_generator.generate_sections(self.knowledge_base.topic, plan)
    
    def run_article_polishing_module(self, article):
        return self.article_polisher.polish_article(self.knowledge_base.topic, article)




from knowledge_storm.storm_wiki.engine import Engine
from ollama_interface import generer_texte

class MyEngine(Engine):
    def __init__(self, lm_configs):
        super().__init__(lm_configs)
        self.topic = None

    def run(self, topic, **kwargs):
        self.topic = topic
        return self.run_knowledge_curation_module(**kwargs)

    def run_knowledge_curation_module(self, **kwargs):
        prompt = f"Sujet : {self.topic}"
        texte = generer_texte(prompt)
        return texte

    def run_outline_generation_module(self, **kwargs):
        pass  # Implémentation future

    def run_article_generation_module(self, **kwargs):
        pass  # Implémentation future

    def run_article_polishing_module(self, **kwargs):
        pass  # Implémentation future
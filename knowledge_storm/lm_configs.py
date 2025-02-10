from abc import ABC

class LMConfigs(ABC):
    def __init__(self):
        self.knowledge_curation_module_lm = None  # Modèle pour la curation de connaissances
        self.outline_generation_module_lm = None # Modèle pour la génération de plans
        self.article_generation_module_lm = None  # Modèle pour la génération d'articles
        self.article_polishing_module_lm = None  # Modèle pour le polissage d'articles

    # Ajoutez ici des méthodes pour configurer les modèles, par exemple :
    def set_knowledge_curation_module_lm(self, model):
        self.knowledge_curation_module_lm = model

    # ... (des méthodes similaires pour les autres modules)

class STORMWikiLMConfigs(LMConfigs):
    def __init__(self):
        super().__init__() # Important d'appeler le constructeur parent !
        # Ajoutez ici des initialisations spécifiques à STORMWikiLMConfigs, si nécessaire.

    # Exemple de méthode spécifique (vous pouvez en ajouter d'autres)
    def init_openai_model(self, openai_api_key, openai_type="openai"):
        # Cette méthode est dépréciée, mais fournie pour compatibilité.
        # Il est préférable de configurer les modèles individuellement avec des setters.
        pass # À adapter si vous souhaitez l'utiliser

    # ... (d'autres méthodes spécifiques à STORMWikiLMConfigs)
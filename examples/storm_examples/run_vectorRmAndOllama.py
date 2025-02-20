import os
from argparse import ArgumentParser
from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.rm import VectorRM
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
from knowledge_storm.utils import load_api_key, QdrantVectorStoreManager
import requests
import litellm

MODELE_OLLAMA = "llama3.2:latest"
PORT_OLLAMA = 11434


def generer_texte(prompt, model=MODELE_OLLAMA, port=PORT_OLLAMA, max_tokens=2000, temperature=0.7):
    print(f"Appel à generer_texte avec prompt: {prompt[:50]}...")
    url = f"http://localhost:{port}/api/generate"
    headers = {"Content-Type": "application/json"}
    
    # Forcer le modèle à répondre en JSON
    prompt = f"Réponds uniquement en JSON bien formé. Assure-toi que la réponse inclut 'done': true à la fin. Voici la question : {prompt}"

    
    data = {
        "model": model,
        "prompt": prompt,
        "format": "json",  # Demande JSON
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        print(f"Réponse complète d'Ollama : {result}")  # Pour vérifier la structure complète de la réponse
        
        if "response" in result:
            print("Réponse générée avec succès.")
            print(f"Réponse d'Ollama: {result['response']}")
            return result["response"]
        elif "error" in result:
            print(f"Erreur d'Ollama: {result['error']}")
            raise Exception(f"Erreur d'Ollama: {result['error']}")
        else:
            print("Réponse JSON inattendue d'Ollama:", result)
            raise Exception(f"Réponse JSON inattendue d'Ollama: {result}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête à Ollama: {e}")
        raise Exception(f"Erreur de requête à Ollama: {e}")



class OllamaWrapper:
    def __init__(self, model=MODELE_OLLAMA, temperature=0.7, max_tokens=500):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = {"temperature": temperature, "max_tokens": max_tokens}

    def __call__(self, prompt, **kwargs):
        self.kwargs.update(kwargs)
        max_tokens = self.kwargs.get("max_tokens", self.max_tokens)
        temperature = self.kwargs.get("temperature", self.temperature)
        print(f"Appel à OllamaWrapper avec prompt: {prompt[:50]}...")
        return generer_texte(
            prompt,
            model=self.model,
            port=PORT_OLLAMA,
            max_tokens=max_tokens,
            temperature=temperature,
        )


litellm.custom_llm_provider = OllamaWrapper()


class DummyCallbackHandler(BaseCallbackHandler):
    def on_identify_perspective_start(self):
        pass

    def on_identify_perspective_end(self, perspectives):
        pass

    def on_information_gathering_start(self):
        pass

    def on_information_gathering_end(self):
        pass

def main(args):
    # Charger la clé API
    load_api_key(toml_file_path=r"examples\storm_examples\secrets.toml")

    # Initialiser les configurations LM
    engine_lm_configs = STORMWikiLMConfigs()

    ollama_kwargs = {
        "model": args.model,
        "port": args.port,
        "url": args.url,
        "stop": ("\n\n---",),
    }

    conv_simulator_lm = OllamaWrapper(max_tokens=500)
    question_asker_lm = OllamaWrapper(max_tokens=500)
    outline_gen_lm = OllamaWrapper(max_tokens=400)
    article_gen_lm = OllamaWrapper(max_tokens=700)
    article_polish_lm = OllamaWrapper(max_tokens=4000)

    engine_lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    engine_lm_configs.set_question_asker_lm(question_asker_lm)
    engine_lm_configs.set_outline_gen_lm(outline_gen_lm)
    engine_lm_configs.set_article_gen_lm(article_gen_lm)
    engine_lm_configs.set_article_polish_lm(article_polish_lm)

    # Initialiser les arguments du moteur
    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    # Créer/mettre à jour le VectorDB avec les documents du fichier CSV
    QdrantVectorStoreManager.create_or_update_vector_store(
        collection_name=args.collection_name,
        vector_db_mode=args.vector_db_mode,
        file_path=args.csv_file_path,
        content_column="content",  # Adjust column name if needed
        url_column="url",  # Adjust column name if needed
        vector_store_path=args.offline_vector_db_dir,
        url=args.online_vector_db_url,
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.embed_batch_size,
    )

    # Configurer VectorRM pour récupérer les informations de vos données
    rm = VectorRM(
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        device=args.device,
        k=engine_args.search_top_k,
    )

    # Initialiser le VectorDB, en ligne ou hors ligne
    if args.vector_db_mode == "offline":
        rm.init_offline_vector_db(vector_store_path=args.offline_vector_db_dir)
    elif args.vector_db_mode == "online":
        rm.init_online_vector_db(
            url=args.online_vector_db_url, api_key=os.getenv("QDRANT_API_KEY")
        )

    # Initialiser le STORM Wiki Runner
    runner = STORMWikiRunner(engine_args, engine_lm_configs, rm)

    # Créer une instance de DummyCallbackHandler
    callback_handler = DummyCallbackHandler()

    # Exécuter le pipeline
    topic = input("Topic: ")
    print(f"Topic entré: {topic}")

    runner.run(
        topic=topic,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
        callback_handler=callback_handler,
    )
   

    # Exécute les modules et capture les sorties
    research_results = runner.run_knowledge_curation_module(callback_handler=callback_handler)

    # Sauvegarde les informations extraites dans un fichier JSON
    research_results.dump_url_to_info("research_results.json")  # Ajout ici

    outline = runner.run_outline_generation_module(information_table=research_results, callback_handler=callback_handler)
    article = runner.run_article_generation_module(outline=outline, information_table=research_results, callback_handler=callback_handler)
    polished_article = runner.run_article_polishing_module(draft_article=article)


    # Affiche les résultats
    print("=== Résultat de la recherche ===")
    print(research_results.to_dict())  # Affiche le contenu de StormInformationTable

    print("=== Résultat du plan ===")
    print(outline.to_string())  # Affiche le contenu de StormArticle

    print("=== Résultat de la génération de l'article ===")
    print(article.to_string())  # Affiche le contenu de StormArticle

    print("=== Résultat du polissage ===")
    print(polished_article.to_string())  # Affiche le contenu de StormArticle

    runner.post_run()

    summary_content = runner.summary()
    print("Contenu du résumé avant d'écrire le fichier:")
    print(summary_content)

if __name__ == "__main__":
    parser = ArgumentParser()
    # arguments globaux
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/ollama_retrieval",
        help="Répertoire pour stocker les sorties.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=int,
        default=3,
        help="Nombre maximal de threads à utiliser. La partie de recherche d'informations et la génération d'articles"
        "peuvent être accélérées en utilisant plusieurs threads. Réduisez-le si vous obtenez "
        '"Exceed rate limit" lors de l\'appel de l\'API LM.',
    )
    # fournir un corpus local et configurer le VectorDB
    parser.add_argument(
        "--collection-name",
        type=str,
        default="my_documents",
        help="Le nom de la collection pour le stockage de vecteurs.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-m3",
        help="Le modèle d'embedding pour le stockage de vecteurs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="L'appareil utilisé pour exécuter le modèle de récupération (mps, cuda, cpu, etc.).",
    )
    parser.add_argument(
        "--vector-db-mode",
        type=str,
        choices=["offline", "online"],
        help="Le mode du stockage de vecteurs Qdrant (hors ligne ou en ligne).",
    )
    parser.add_argument(
        "--offline-vector-db-dir",
        type=str,
        default="./vector_store",
        help="Si le mode hors ligne est utilisé, fournir le répertoire pour stocker le stockage de vecteurs.",
    )
    parser.add_argument(
        "--online-vector-db-url",
        type=str,
        help="Si le mode en ligne est utilisé, fournir l'URL du serveur Qdrant.",
    )
    parser.add_argument(
        "--csv-file-path",
        type=str,
        default=None,
        help="Le chemin du corpus de documents personnalisé au format CSV. Le fichier CSV doit inclure "
        "les colonnes content, title, url et description.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Taille du lot pour l'embedding des documents dans le fichier CSV.",
    )
    # étape du pipeline
    parser.add_argument(
        "--do-research",
        action="store_true",
        help="Si True, simuler une conversation pour rechercher le sujet; sinon, charger les résultats.",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="Si True, générer un plan pour le sujet; sinon, charger les résultats.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="Si True, générer un article pour le sujet; sinon, charger les résultats.",
    )
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="Si True, polir l'article en ajoutant une section de résumé et (optionnellement) en supprimant "
        "le contenu en double.",
    )
    # hyperparamètres pour l'étape de pré-écriture
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=3,
        help="Nombre maximal de questions dans la conversation de questionnement.",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="Nombre maximal de perspectives à considérer dans le questionnement guidé par les perspectives.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=3,
        help="Les k meilleurs résultats de recherche à considérer pour chaque requête de recherche.",
    )
    # hyperparamètres pour l'étape d'écriture
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=3,
        help="Les k meilleures références collectées pour chaque titre de section.",
    )
    parser.add_argument(
        "--remove-duplicate",
        action="store_true",
        help="Si True, supprimer le contenu en double de l'article.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost",
        help="Ollama URL.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Ollama port.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:latest",
        help="Ollama model.",
    )

    main(parser.parse_args())
# MAINTiDOC — Étape 1 : RAG textuel

Assistant documentaire pour techniciens de maintenance : questions techniques,
réponses sourcées avec citations vérifiables, filtrage des documents par groupe d'accès.
Il ne pilote aucun équipement et ne remplace pas un technicien habilité.

## Architecture

PDF → PyMuPDF (extraction par page) → passages + métadonnées → BGE-M3 → Qdrant
Question → API FastAPI (jeton → groupes) → recherche filtrée → Qwen (Ollama) →
vérification des citations → réponse sourcée / abstention

## Installation (Windows, Python 3.11)

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
    ollama pull qwen2.5:7b

## Configuration

Copier `.env.example` en `.env` et renseigner deux jetons aléatoires :

    python -c "import secrets; print(secrets.token_urlsafe(24))"

## Utilisation

1. Déposer des PDF dans `data/docs/` (non versionnés)
2. `python ingest.py` (arrêter l'API avant : Qdrant local n'accepte qu'un processus)
3. `uvicorn api:app --port 8000`
4. `streamlit run app.py` (dans un second terminal)

## Fichiers

- `ingest.py` : extraction, découpage, métadonnées, embeddings, indexation
- `rag.py` : recherche filtrée, génération, vérification des citations
- `api.py` : API FastAPI, authentification par jeton, journalisation minimale
- `app.py` : interface Streamlit
- `prompts/` : prompts versionnés (answer_v1 à answer_v6)

## Limites connues

- La vérification des citations est un proxy de similarité, pas une preuve d'entailment.
- Des réponses sous forme de listes courtes peuvent être signalées comme partielles.
- Les jetons de démonstration ne remplacent pas une vraie authentification.
- Latence mesurée sur RTX 3050 portable avec Qwen 7B partiellement sur CPU : élevée à froid.
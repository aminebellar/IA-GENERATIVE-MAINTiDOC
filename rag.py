from typing import Literal, Optional
from pathlib import Path
import re, yaml, ollama
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

COLLECTION = "maintidoc_text"
LLM = "qwen2.5:7b"
PROMPT_VERSION = "answer_v6"
MIN_SCORE = 0.40          # seuil de récupération
SUPPORT_MIN = 0.45        # seuil de soutien phrase/passage : à calibrer
TOP_K = 5

class Citation(BaseModel):
    ref: int
    document_id: str
    page: int
    element_id: str

class RagAnswer(BaseModel):
    status: Literal["answered", "needs_clarification", "no_source"]
    answer: str
    citations: list[Citation] = []
    clarification_question: Optional[str] = None
    retrieved: list[str] = []
    support_trace: list[str] = []
    raw_llm: Optional[str] = None
    prompt_version: str = PROMPT_VERSION
    model: str = LLM

_embedder = SentenceTransformer("BAAI/bge-m3", device="cuda").half()
_client = QdrantClient(path="./qdrant_data")
_system = yaml.safe_load(
    Path(f"prompts/{PROMPT_VERSION}.yaml").read_text(encoding="utf-8"))["system"]

def retrieve(question, user_groups):
    vec = _embedder.encode(question, normalize_embeddings=True).tolist()
    hits = _client.query_points(
        COLLECTION, query=vec, limit=TOP_K,
        query_filter=Filter(must=[FieldCondition(
            key="allowed_groups", match=MatchAny(any=user_groups))]),
    ).points
    return [h for h in hits if h.score >= MIN_SCORE]

def call_llm(user_msg):
    resp = ollama.chat(
        model=LLM,
        messages=[{"role": "system", "content": _system},
                  {"role": "user", "content": user_msg}],
        options={"temperature": 0, "num_ctx": 4096},
    )
    return resp["message"]["content"].strip()

def split_sentences(text):
    # ne coupe pas avant un marqueur [n] qui suit un point
    parts = re.split(r"(?<=[.!?])\s+(?!\[\d+\])|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]

def verify_citations(text, hits):
    """Garde les phrases dont au moins une citation est proche du passage cité."""
    kept_sentences, kept_refs, trace = [], set(), []
    for s in split_sentences(text):
        refs = sorted({int(n) for n in re.findall(r"\[(\d+)\]", s)
                       if 1 <= int(n) <= len(hits)})
        clean = re.sub(r"\s*\[\d+\]", "", s).strip(" :-")
        if not refs or len(clean) < 25:
            continue
        s_vec = _embedder.encode(clean, normalize_embeddings=True).astype("float32")
        ok = []
        for r in refs:
            p_vec = _embedder.encode(hits[r-1].payload["text"],
                                     normalize_embeddings=True).astype("float32")
            score = float(s_vec @ p_vec)
            verdict = "ok" if score >= SUPPORT_MIN else "rejeté"
            trace.append(f"[{r}] {score:.2f} {verdict} : {clean[:60]}")
            if score >= SUPPORT_MIN:
                ok.append(r)
        if ok:
            kept_sentences.append(clean + " " + "".join(f"[{r}]" for r in ok))
            kept_refs.update(ok)
    return kept_sentences, sorted(kept_refs), trace

def ask(question, user_groups=("technicien",)):
    hits = retrieve(question, list(user_groups))
    retrieved = [f"{h.payload['document_id']} p.{h.payload['page']} ({h.score:.2f})" for h in hits]
    no_source = RagAnswer(
        status="no_source", retrieved=retrieved,
        answer="Les sources accessibles ne permettent pas de répondre de façon sourcée.")
    if not hits:
        return no_source

    context = "\n\n".join(
        f"[{i}] (document {h.payload['document_id']}, page {h.payload['page']})\n{h.payload['text']}"
        for i, h in enumerate(hits, start=1))
    prompt = f"EXTRAITS :\n{context}\n\nQUESTION : {question}"

    def is_special(t):
        return "AUCUNE_SOURCE" in t or t.upper().startswith("CLARIFICATION:")

    def too_thin(t):
        # pas de marqueur, ou texte quasi vide une fois les marqueurs retirés
        return (not re.search(r"\[\d+\]", t)) or len(re.sub(r"\[\d+\]", "", t).strip()) < 25

    text = call_llm(prompt)
    if not is_special(text) and too_thin(text):
        text = call_llm(prompt + "\n\nRappel : rédige une réponse en phrases complètes "
                                 "et termine chaque phrase par un numéro d'extrait entre "
                                 "crochets, par exemple [1]. Ne réponds pas avec des numéros seuls.")

    if text.upper().startswith("CLARIFICATION:"):
        return RagAnswer(status="needs_clarification", retrieved=retrieved,
                         answer="Une précision est nécessaire pour répondre.",
                         clarification_question=text.split(":", 1)[1].strip())

    if "AUCUNE_SOURCE" in text:
        no_source.raw_llm = text
        return no_source

    sentences, refs, trace = verify_citations(text, hits)
    if not refs:
        no_source.raw_llm = text
        no_source.support_trace = trace
        return no_source

    total = len([s for s in split_sentences(text) if re.search(r"\[\d+\]", s)])
    dropped = total - len(sentences)
    answer = "\n".join(sentences)
    if dropped > 0:
        answer += (f"\n\n(Réponse partielle : {dropped} élément(s) de la réponse générée "
                   f"n'ont pas pu être rattachés aux sources et ont été retirés.)")

    citations = [Citation(ref=i, document_id=hits[i-1].payload["document_id"],
                          page=hits[i-1].payload["page"],
                          element_id=hits[i-1].payload["element_id"]) for i in refs]
    return RagAnswer(status="answered", answer=answer,
                     citations=citations, retrieved=retrieved, support_trace=trace)
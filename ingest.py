import sys, uuid
from pathlib import Path
import pymupdf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "maintidoc_text"
DOCS_DIR = Path("data/docs")
DEFAULT_GROUPS = ("technicien", "responsable")
PERMISSIONS = {"feuille de route": ("responsable",)}   # mot-clé du nom de fichier -> groupes

def groups_for(filename):
    for key, groups in PERMISSIONS.items():
        if key in filename.lower():
            return groups
    return DEFAULT_GROUPS

def chunk_text(text, size=1000, overlap=150):
    """Découpe un texte en passages de ~size caractères avec chevauchement."""
    text = " ".join(text.split())          # nettoie les retours à la ligne
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def extract_passages(pdf_path, version="1.0", equipment_type="general",
                     allowed_groups=("technicien", "responsable")):
    doc_id = pdf_path.stem
    with pymupdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            for i, chunk in enumerate(chunk_text(page.get_text())):
                if len(chunk.strip()) < 30:   # ignore les passages vides
                    continue
                yield {
                    "text": chunk,
                    "document_id": doc_id,
                    "version": version,
                    "page": page_num,
                    "element_id": f"{doc_id}_p{page_num}_c{i}",
                    "equipment_type": equipment_type,
                    "allowed_groups": list(allowed_groups),
                }

def main():
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.half()                              # fp16 pour ménager la VRAM
    client = QdrantClient(path="./qdrant_data")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        passages = list(extract_passages(pdf_path, allowed_groups=groups_for(pdf_path.name)))
        vectors = model.encode([p["text"] for p in passages],
                               batch_size=16, normalize_embeddings=True)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, p["element_id"])),
                vector=v.tolist(),
                payload=p,
            )
            for p, v in zip(passages, vectors)
        ]
        client.upsert(COLLECTION, points=points)
        print(f"{pdf_path.name} : {len(points)} passages indexés")

if __name__ == "__main__":
    main()
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

question = sys.argv[1]
user_groups = sys.argv[2:] or ["technicien"]

model = SentenceTransformer("BAAI/bge-m3", device="cuda")
client = QdrantClient(path="./qdrant_data")

vec = model.encode(question, normalize_embeddings=True).tolist()
hits = client.query_points(
    "maintidoc_text",
    query=vec,
    query_filter=Filter(must=[
        FieldCondition(key="allowed_groups", match=MatchAny(any=user_groups))
    ]),
    limit=5,
).points

for h in hits:
    p = h.payload
    print(f"[{h.score:.3f}] {p['document_id']} p.{p['page']}\n  {p['text'][:150]}...\n")
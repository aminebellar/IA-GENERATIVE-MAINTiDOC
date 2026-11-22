import torch, pymupdf, qdrant_client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

print("CUDA :", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("PyMuPDF :", pymupdf.__doc__.split("\n")[0])
print("Qdrant :", QdrantClient(path="./qdrant_data").get_collections())
SentenceTransformer("BAAI/bge-m3")
print("BGE-M3 OK")
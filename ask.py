import sys
from rag import ask
r = ask(sys.argv[1], sys.argv[2:] or ["technicien"])
print(r.model_dump_json(indent=2, exclude_none=True))
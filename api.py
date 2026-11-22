import os, time, logging, secrets, threading
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from rag import ask

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("maintidoc")
DEBUG = os.getenv("DEBUG") == "1"

# Identités de démonstration : jeton -> (nom, groupes). Les jetons viennent de .env.
USERS = []
for env_name, name, groups in (
    ("TOKEN_TECHNICIEN", "technicien_demo", ["technicien"]),
    ("TOKEN_RESPONSABLE", "responsable_demo", ["responsable"]),
):
    token = os.getenv(env_name, "")
    if token:                                   # un jeton vide n'ouvre aucun accès
        USERS.append({"token": token, "name": name, "groups": groups})

app = FastAPI(title="MAINTiDOC API", version="0.1")
_lock = threading.Lock()                        # une requête à la fois (GPU limité)

def current_user(authorization: str = Header(default="")):
    token = authorization.removeprefix("Bearer ").strip()
    for user in USERS:
        if secrets.compare_digest(token, user["token"]):
            return user
    raise HTTPException(status_code=401, detail="Jeton invalide ou absent")

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_endpoint(req: AskRequest, user=Depends(current_user)):
    t0 = time.time()
    with _lock:
        result = ask(req.question, user["groups"])
    data = result.model_dump(exclude_none=True)
    if not DEBUG:                               # traces internes réservées au debug
        data.pop("raw_llm", None)
        data.pop("support_trace", None)
    # journalisation minimale : métadonnées uniquement, jamais le contenu
    log.info("user=%s status=%s citations=%d latency=%.1fs",
             user["name"], result.status, len(result.citations), time.time() - t0)
    return data
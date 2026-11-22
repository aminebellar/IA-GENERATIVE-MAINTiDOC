import os, requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("API_URL", "http://127.0.0.1:8000")
TOKENS = {"technicien": os.getenv("TOKEN_TECHNICIEN", ""),
          "responsable": os.getenv("TOKEN_RESPONSABLE", "")}

st.set_page_config(page_title="MAINTiDOC", page_icon="🛠️")
st.title("MAINTiDOC")
st.caption("Assistant documentaire pour la maintenance : réponses sourcées uniquement.")

role = st.sidebar.selectbox("Profil de démonstration", list(TOKENS))
st.sidebar.caption("Les documents accessibles dépendent du profil choisi.")

question = st.text_area("Votre question technique", height=100)

if st.button("Envoyer", type="primary") and question.strip():
    with st.spinner("Recherche des sources et génération de la réponse..."):
        try:
            r = requests.post(f"{API}/ask", json={"question": question},
                              headers={"Authorization": f"Bearer {TOKENS[role]}"},
                              timeout=300)
        except requests.RequestException as e:
            st.error(f"API injoignable : {e}")
            st.stop()
    if r.status_code != 200:
        st.error(f"Erreur {r.status_code} : {r.text}")
        st.stop()

    data = r.json()
    status = data["status"]
    if status == "answered":
        st.success("Réponse sourcée")
        st.markdown(data["answer"].replace("\n", "\n\n"))
    elif status == "needs_clarification":
        st.info(data.get("clarification_question", "Une précision est nécessaire."))
    else:
        st.warning(data["answer"])

    if data.get("citations"):
        with st.expander(f"Sources ({len(data['citations'])})", expanded=True):
            for c in data["citations"]:
                st.markdown(f"**[{c['ref']}]** {c['document_id']}, page {c['page']}  \n"
                            f"`{c['element_id']}`")
    st.caption(f"Modèle : {data['model']} · prompt : {data['prompt_version']}")
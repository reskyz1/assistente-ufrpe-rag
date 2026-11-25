"""Interface local de consulta com fontes visíveis."""

import os

import streamlit as st
from dotenv import load_dotenv

from rag_ufrpe.corpus import ROOT, SUPPORTED_SEMESTERS, sources

load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

st.set_page_config(page_title="Assistente acadêmico UFRPE", page_icon="📚", layout="centered")
st.title("Consulte as normas acadêmicas")
st.write("Matrícula e trancamento na graduação presencial da UFRPE.")
st.caption("Protótipo acadêmico independente. Base documental conferida em 23/11/2026. Confira os trechos antes de seguir uma orientação.")

with st.sidebar:
    st.subheader("Sua consulta")
    semester_label = st.selectbox("Semestre para consultar prazos", ["Informar na pergunta", *SUPPORTED_SEMESTERS])
    mode = st.selectbox("Formato da resposta", ["Resposta com fontes", "Consultar somente os trechos"])
    st.subheader("Documentos da base")
    for source in sources():
        st.link_button(source["title"], source["page_url"])

st.write("Você pode perguntar “Em quanto tempo o trancamento de uma disciplina é efetivado?” ou “Qual o prazo de matrícula em 2026.2?”.")
with st.form("question"):
    question = st.text_input("Qual é sua dúvida?", max_chars=1000)
    submitted = st.form_submit_button("Consultar", type="primary")


@st.cache_resource
def get_retriever():
    from rag_ufrpe.retrieval import Retriever
    return Retriever()


def render_result(result):
    st.subheader("Resposta")
    st.write(result["answer"])
    if result["mode"] in {"generation_error", "invalid_citations"}:
        st.info("A consulta retornou fontes para leitura. Tente novamente após conferir a configuração do gerador.")
    for number, source in enumerate(result["sources"], start=1):
        pages = ", ".join(map(str, source["pages"]))
        with st.expander(f"[{number}] {source['title']} · página(s) {pages}", expanded=number == 1):
            st.write(source["text"])
            st.link_button("Abrir documento na fonte oficial", source["url"] + f"#page={source['pages'][0]}")
    if result["sources"]:
        st.caption("As citações indicam os documentos usados. A correção de cada afirmação ainda exige conferência.")


if submitted:
    from rag_ufrpe.generation import answer
    from rag_ufrpe.retrieval import retrieve

    st.session_state.pop("result", None)
    semester = None if semester_label == "Informar na pergunta" else semester_label
    try:
        with st.spinner("Consultando os documentos…"):
            retrieval = retrieve(question, semester, get_retriever)
            backend = "sources" if mode == "Consultar somente os trechos" else os.getenv("GENERATION_BACKEND", "local")
            st.session_state["result"] = answer(retrieval, question, backend)
    except FileNotFoundError:
        st.error("Prepare a base com python -m rag_ufrpe preparar e tente novamente.")
    except (ImportError, OSError, RuntimeError, ValueError):
        st.error("Não foi possível carregar a busca. Confira a instalação e a conexão no primeiro download do modelo.")
if "result" in st.session_state:
    render_result(st.session_state["result"])

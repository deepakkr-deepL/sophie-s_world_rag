import streamlit as st

from src.config.config import Config
from src.chains.serve_chain import get_rag_chain


st.set_page_config(page_title="Sophie's World RAG", layout="wide")

st.title("📘 Sophie's World – RAG")


@st.cache_resource
def load_chain():
    return get_rag_chain()


chain = load_chain()

st.sidebar.markdown("### ⚙️ Config")
st.sidebar.write("File:", Config.FILE_NAME)
st.sidebar.write("Embedding:", Config.EMBEDDING_MODEL)
st.sidebar.write("LLM:", Config.LLM_MODEL)

question = st.text_input("Ask a question")

if st.button("Submit"):
    if question.strip():
        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke(question)

                st.subheader("🧠 Answer")
                st.write(answer)

            except Exception as e:
                st.error(str(e))

from langchain_classic.retrievers import MultiQueryRetriever, ParentDocumentRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.prompts.prompts import get_philosophy_rag_prompt
from src.config.config import Config
from src.vector_store.vector_store import get_vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def format_docs(docs):
    return iter("\n\n".join([
        f"Page {doc.metadata.get('page', '?') + 1} • {doc.page_content.strip()}"
        for doc in docs
    ]))


def create_sophies_world_rag_chain_similarity_retriever():
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Config.RETRIEVER_K}
    )
    # retriever = cross_encoder_reranker(sim_retriever)
    llm = ChatOllama(
        model=Config.LLM_MODEL,
        temperature=0.2,
        num_ctx=8192,
    )

    prompt = get_philosophy_rag_prompt()

    chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


def create_sophies_world_rag_chain_with_mmr_retriever():
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": Config.RETRIEVER_K,
            "fetch_k": 2,
            "lambda_mult": 0.7
        }
    )

    llm = ChatOllama(
        model=Config.LLM_MODEL,
        temperature=0.2,
        num_ctx=8192,
    )
    prompt = get_philosophy_rag_prompt()

    chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


def create_sophies_world_rag_chain_with_multi_query_retriever():
    vectorstore = get_vectorstore()
    llm = ChatOllama(
        model=Config.LLM_MODEL,
        temperature=0.2,
        num_ctx=8192,
    )

    MULTI_QUERY_PROMPT = PromptTemplate.from_template(
        """You are an expert at reformulating questions.
    Given the original question, generate 4 different versions that would help
    retrieve more relevant documents from a philosophical novel knowledge base.

    Make them semantically similar but vary wording, perspective, and keywords.

    Original question: {question}

    Return only the 4 questions, one per line, no numbering, no extra text."""
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=llm,
        include_original=True,
        prompt=MULTI_QUERY_PROMPT
    )

    prompt = get_philosophy_rag_prompt()

    chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain

# def create_sophies_world_rag_chain_with_parent_child_retriever():
#     """
#     Parent-Child retriever ke saath full RAG chain banata hai.
#     Vectorstore ko get_vectorstore() se leta hai (existing load priority).
#     """
#     # 1. Existing vector store load karo (child chunks ke saath)
#     vectorstore = get_vectorstore(force_recreate=False)  # tumhara function
#
#     # 2. Child splitter (search ke liye chhote chunks)
#     # Note: Agar vectorstore already bana hai aur usme different size ke chunks hain,
#     # to child_splitter ko match karna padega ya vectorstore recreate karna padega
#     child_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=400,  # ← tumhare latest discussion ke hisaab se
#         chunk_overlap=50,
#         add_start_index=True
#     )
#
#     # 3. Parent splitter (LLM ko bada context dene ke liye)
#     parent_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2000,
#         chunk_overlap=200
#     )
#
#     # 4. Doc store for full parent documents
#     docstore = InMemoryStore()
#
#     # 5. Parent-Child Retriever create karo
#     retriever = ParentDocumentRetriever(
#         vectorstore=vectorstore,
#         docstore=docstore,
#         child_splitter=child_splitter,
#         parent_splitter=parent_splitter,
#     )
#
#     # 6. LLM setup (same as MMR version)
#     llm = ChatOllama(
#         model=Config.LLM_MODEL,
#         temperature=0.2,
#         num_ctx=8192,
#     )
#
#     # 7. Prompt (tumhara philosophy wala)
#     prompt = get_philosophy_rag_prompt()
#
#     # 9. Full LCEL chain (MMR version ke bilkul same structure)
#     chain = (
#             {
#                 "context": retriever | RunnableLambda(format_docs),
#                 "question": RunnablePassthrough()
#             }
#             | prompt
#             | llm
#             | StrOutputParser()
#     )
#
#     return chain

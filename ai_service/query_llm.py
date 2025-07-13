from typing import Iterator

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank

from .utils.thinking_animation import ThinkingAnimation
from .utils.query import get_llm, generate_prompt, format_docs, process_chunk


def invoke_query(query: str, vectorstore: Chroma) -> str:
    """Execute a query using retrieval-augmented generation (RAG) with document compression.

    Attempts to use LLM listwise reranking for document compression. If compression fails,
    falls back to standard retrieval without compression.

    Args:
        query (str): The user's query string.
        vectorstore (Chroma): The ChromaDB vectorstore containing the documents.

    Returns:
        str: The LLM's response based on the retrieved and compressed documents.
    """
    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 7}
    )

    prompt = generate_prompt()

    try:
        compressor = LLMListwiseRerank.from_llm(llm=llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        rag_chain = (
            {
                "context": compression_retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        # Execute the RAG chain with thinking animation
        with ThinkingAnimation("Thinking"):
            result = rag_chain.invoke(query)
    except AttributeError as e:
        print(
            "No relevant context found for the query. Using standard retrieval without compression."
        )
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        # Execute the RAG chain with thinking animation
        with ThinkingAnimation("Thinking"):
            result = rag_chain.invoke(query)

    return result


def invoke_query_stream(query: str, vectorstore: Chroma) -> Iterator[str]:
    """Execute a query using retrieval-augmented generation (RAG) with streaming output.

    Attempts to use LLM listwise reranking for document compression. If compression fails,
    falls back to standard retrieval without compression. Responses are streamed as they
    are generated.

    Args:
        query (str): The user's query string.
        vectorstore (Chroma): The ChromaDB vectorstore containing the documents.

    Yields:
        str: Streaming chunks of the LLM's response.
    """

    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 7}
    )

    prompt = generate_prompt()

    try:

        compressor = LLMListwiseRerank.from_llm(llm=llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        rag_chain = (
            {
                "context": compression_retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        # Stream the response
        print("Thinking...")
        for chunk in rag_chain.stream(query):
            processed_chunk = process_chunk(chunk)
            if processed_chunk:  # Only yield non-empty chunks
                yield processed_chunk

    except AttributeError as e:
        print(
            "No relevant context found for the query. Using standard retrieval without compression."
        )
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        # Stream the response
        print("🤔 Thinking...")
        for chunk in rag_chain.stream(query):
            processed_chunk = process_chunk(chunk)
            if processed_chunk:  # Only yield non-empty chunks
                yield processed_chunk

from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_huggingface import HuggingFaceEmbeddings

import datetime
from llm.main import get_llm



def format_docs(docs: Iterable[LCDocument]):
    """Format documents with content and source information.
    
    Args:
        docs (Iterable[LCDocument]): An iterable of LangChain Document objects.
        
    Returns:
        str: A formatted string containing document content with source information.
    """
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        
        # Extract source information
        source_info = ""
        if metadata:
            file_path = metadata.get('file_path', 'Unknown')
            chunk_id = metadata.get('chunk_id', 'Unknown')
            
            source_info = f"[Source: {file_path}, Chunk: {chunk_id}]"
        
        formatted_docs.append(f"{content}\n{source_info}")
    
    return "\n\n".join(formatted_docs)

def invoke_query(query: str, vectorstore: Chroma, embeddings: HuggingFaceEmbeddings) -> str:
    """Execute a query using retrieval-augmented generation (RAG) with document compression.
    
    Args:
        query (str): The user's query string.
        vectorstore (Chroma): The ChromaDB vectorstore containing the documents.
        embeddings (HuggingFaceEmbeddings): The embedding function used for the vectorstore.

    Returns:
        str: The LLM's response based on the retrieved and compressed documents.
    """
    groq_llm = get_llm()

    retriver = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = PromptTemplate.from_template(
        "Context information is below. Each piece of context includes source information in brackets.\n"
        "---------------------\n{context}\n---------------------\n"
        "Given the context information and not prior knowledge, answer the query. "
        "Only include relevant information from the context."
        f"The current Date and Time is {current_date}. You don't need to state where it was found.\n"
        "If relevant, please include the source file path(s) where the information was found and place that information at the bottom.\n"
        "Use markdown formatting for the answer.\n"
        "Directly answer the question without using a heading like Answer\n"
        "Query: {question}\nAnswer:\n"
    )

    compressor = LLMListwiseRerank.from_llm(llm=groq_llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriver)
    
    rag_chain_compressor = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | groq_llm 
        | StrOutputParser()
    )

    return rag_chain_compressor.invoke(query)

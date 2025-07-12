from typing import Iterable
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain_core.embeddings import Embeddings

import datetime
from llm.model import LLMProvider,LLM
from utils.thinking_animation import ThinkingAnimation

load_dotenv()
LLM_SERVICE = os.getenv('LLM_SERVICE', 'qwen')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'qwen3-32b')



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

def invoke_query(query: str, vectorstore: Chroma, embeddings: Embeddings) -> str:
    """Execute a query using retrieval-augmented generation (RAG) with document compression.
    
    Args:
        query (str): The user's query string.
        vectorstore (Chroma): The ChromaDB vectorstore containing the documents.
        embeddings (HuggingFaceEmbeddings): The embedding function used for the vectorstore.

    Returns:
        str: The LLM's response based on the retrieved and compressed documents.
    """
    llmProvider = LLMProvider(LLM_SERVICE.lower())
    llmService = LLM(llmProvider, LLM_MODEL_NAME)
    llm = llmService.get_llm()

    retriver = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":7})

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = PromptTemplate.from_template(
        "Context information is below. Each piece of context includes source information in brackets.\n"
        "---------------------\n{context}\n---------------------\n"
        "If you find the context to be relevant, use it to answer the question."
        "If you found relevant context, always include the source file path(s) where the information was found and place that information at the bottom. Include the full path, not just the filename. Don't include the corresponding chunk_id."
        "Only give the source file paths for information that you got from the context, not for information that you already know. Never state that you didn't find sources for non context information.\n"
        "Only include relevant information from the context."
        "If you list the source file path of a context information, list it only once, even if it was used multiple times and different chunks where used."
        "If no context information is relevant, don't mention it, if you are not specifically asked to do so.\n"
        "Never answer with one word or a single sentence if you are not specifically asked to do so.\n"
        "If you are asked a question in a specific language, always answer in that language.\n"
        f"The current Date and Time is {current_date}. You don't need to state where it was found.\n"
        "Use markdown formatting for the answer.\n"
        "Query: {question}\nAnswer:\n"
    )


    
    try:
        compressor = LLMListwiseRerank.from_llm(llm=llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriver)
        
        rag_chain = (
            {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm 
            | StrOutputParser()
        )
        # Execute the RAG chain with thinking animation
        with ThinkingAnimation("Thinking"):
            result = rag_chain.invoke(query)
    except AttributeError as e:
        print("No relevant context found for the query. Using standard retrieval without compression.")
        rag_chain = (
            {"context": retriver | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm 
            | StrOutputParser()
        )
        # Execute the RAG chain with thinking animation
        with ThinkingAnimation("Thinking"):
            result = rag_chain.invoke(query)
    
    
    return result

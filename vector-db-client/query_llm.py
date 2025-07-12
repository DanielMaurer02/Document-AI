from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter

from llm.main import get_groq_llm



def format_docs(docs: Iterable[LCDocument]):
    """Format documents with content and source information."""
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

def invoke_query(query: str, vectorstore: Chroma) -> str:
    groq_llm = get_groq_llm()

    retriver = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})

    prompt = PromptTemplate.from_template(
        "Context information is below. Each piece of context includes source information in brackets.\n"
        "---------------------\n{context}\n---------------------\n"
        "Given the context information and not prior knowledge, answer the query. "
        "If relevant, please include the source file path(s) where the information was found.\n"
        "Query: {question}\nAnswer:\n"
    )


    filter_prompt = PromptTemplate.from_template(
        "Given the following document chunk, respond with ONLY 'YES' if it is relevant to the query, or 'NO' if it is not. Respond with a single word: YES or NO.\n\nDocument: {context}\nQuery: {question}\nAnswer:"
    )
    compressor = LLMChainFilter.from_llm(llm=groq_llm, prompt=filter_prompt)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriver)


    rag_chain_compressor = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | groq_llm 
        | StrOutputParser()
    )


    return rag_chain_compressor.invoke(query)

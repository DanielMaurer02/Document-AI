from ai_service.main import DocumentAI

file_path = [
    r"c:\Users\danie\Downloads\20240930 DHBW Zeugnis Daniel Maurer .pdf",
    r"c:\Users\danie\Downloads\MV blanko 1.OG rechts.pdf",
]
doc_ai = DocumentAI("rag")
# doc_ai.delete_collection("rag")  # Clear the collection before adding new documents
doc_ai.add_documents(file_path)

from ai_service.main import DocumentAI

file_path = ["/Users/danie/Downloads/20240930 DHBW Zeugnis Daniel Maurer .pdf"
]
doc_ai = DocumentAI("rag")
#doc_ai.delete_collection("rag")  # Clear the collection before adding new documents
#doc_ai.add_documents(file_path)

r = doc_ai.query("Welche Note hatte ich in meinem Abschlusszeugnis?")
print(r)
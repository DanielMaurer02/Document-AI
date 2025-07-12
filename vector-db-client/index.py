import chromadb

client = chromadb.HttpClient("localhost", 8000)

client.delete_collection("rag")
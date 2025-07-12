import  os
from getpass import getpass

from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from add_documents import add_documents_to_chromadb
from query_llm import invoke_query

load_dotenv()

#os.environ['HUGGINGFACE_API_KEY'] = getpass('HF_TOKEN')

persistent_client = chromadb.HttpClient("localhost", 8000)


 # TODO: save table of already saved documents to avoid duplicates
file_path = ["/Users/danie/Downloads/Rechnung-Herman-Miller-Chairgo.pdf","/Users/danie/Downloads/20240930 DHBW Zeugnis Daniel Maurer .pdf"]


HF_EMBED_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL_ID)

vectorstore = Chroma(
    collection_name="rag",
    embedding_function=embeddings,
    client=persistent_client,
    collection_metadata={"hnsw:space": "cosine"},
)

#add_documents_to_chromadb(file_path, vectorstore)

# TODO: Save the Answers temporarily to another collection to be able to use them
res = invoke_query("Welche Note hatte Daniel in seiner BA und was für einen Bürostuhl hat er gekauft?", vectorstore)

print(res)
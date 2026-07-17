from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("Module_1/python.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name = "python_docs",
    embedding_function = embeddings,
    persist_directory="./chroma_db"
)

# Add Documents
vector_store.add_documents(chunks)

# similarity Search
results = vector_store.similarity_search("What is Python?",k=3)

print(results[0].page_content)
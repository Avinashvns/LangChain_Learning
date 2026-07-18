from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("Module_2/python.pdf")
documents = loader.load()

# split
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=30
)

chunks = splitter.split_documents(documents)

# Embedding
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Create Faiss
vector_db = FAISS.from_documents(chunks , embeddings)

# Save Database
vector_db.save_local("Module_2/vector_store")

print("Vector Database Created Sucessfully")
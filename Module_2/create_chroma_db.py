from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("Module_2/python.pdf")
documents = loader.load()

# Split Documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap= 30
)

chunks = splitter.split_documents(documents)

# Create Embedding Model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create chroma DB
vector_db = Chroma(
    collection_name="python_docs",
    embedding_function=embeddings,
    persist_directory="Module_2/chroma_db"
) 

# Add Documents
vector_db.add_documents(chunks)
print("Chroma Database Created Sucessfully")
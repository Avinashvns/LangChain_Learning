from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Step 1 : Load All PDFs
documents = []
pdf_folder = Path("Module_2/data")
pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF found inside data folder.")

print(f"Total PDF : {len(pdf_files)}")

for pdf in pdf_files:
    loader = PyPDFLoader(str(pdf))
    docs = loader.load()
    documents.extend(docs)

print(f"\nTotal Pages : {len(documents)}")

# Step 2 : Split Documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size= 300,
    chunk_overlap = 30
)

chunks = splitter.split_documents(documents)
print(f"Total Chunks : {len(chunks)}")

# Step 3 : Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Step 4 : Create FAISS
vector_db = FAISS.from_documents(chunks, embeddings)

# Step 5 : Save FAISS
vector_db.save_local("Module_2/multiple_vector_store")

print("\nVector Database Created Successfully")
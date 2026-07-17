from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------
# Step 1 : Load PDF
# ---------------------------------

loader = PyPDFLoader("Module_1/python.pdf")
documents = loader.load()

print(f"Total Pages : {len(documents)}")

# ---------------------------------
# Step 2 : Split Documents
# ---------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks : {len(chunks)}")

# ---------------------------------
# Step 3 : Create Embedding Model
# ---------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ---------------------------------
# Step 4 : Create FAISS Database
# ---------------------------------

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("✅ FAISS Database Created Successfully")

# ---------------------------------
# Step 5 : Save FAISS Database
# ---------------------------------

vector_store.save_local("Module_1/faiss_db")

print("✅ FAISS Database Saved Successfully")

# ---------------------------------
# Step 6 : Similarity Search
# ---------------------------------

query = "What is Python?"

results = vector_store.similarity_search(
    query=query,
    k=3
)

print("\n========== SEARCH RESULTS ==========\n")

for i, doc in enumerate(results, start=1):

    print("=" * 70)
    print(f"Result {i}")
    print("=" * 70)

    print("Metadata:")
    print(doc.metadata)

    print("\nContent:")
    print(doc.page_content)
    print()
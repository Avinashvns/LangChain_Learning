from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("Module_1/python.pdf")
documents = loader.load()

# Text Splits
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents)

# Embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Vector Store
vector_store = FAISS.from_documents(chunks,embeddings)

# Retriever
retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3})

# Search
results = retriever.invoke("What is Python?")

# print
for doc in results:
    print(doc.page_content)
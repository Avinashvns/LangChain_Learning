from langchain_ollama import ChatOllama , OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain

# Load PDF
loader = PyPDFLoader("Module_1/python.pdf")
documents = loader.load()

# Test Split
splitters = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap = 50
)
chunks = splitters.split_documents(documents)

# Embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Faiss
vector_store = FAISS.from_documents(chunks,embeddings)

# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k" : 3}
)

# LLM
llm = ChatOllama(model="llama3.2")

# Prompts
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
reply with "I don't know."

Context:
{context}
""",
        ),
        (
            "human",
            "{input}"
        ),
    ]
)

# Document Chain
document_chain = create_stuff_documents_chain(llm,prompt)

# Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Ask Question
response = retrieval_chain.invoke({"input" : "What is Python?"})

print("\n========== ANSWER ==========\n")
print(response["answer"])
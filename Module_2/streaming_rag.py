from langchain_community.vectorstores import FAISS

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from langchain_core.prompts import ChatPromptTemplate

# =====================================
# Embedding Model
# =====================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# =====================================
# Load FAISS
# =====================================

vector_db = FAISS.load_local(
    "Module_2/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# =====================================
# Retriever
# =====================================

retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)

# =====================================
# LLM
# =====================================

llm = ChatOllama(
    model="llama3.2"
)

# =====================================
# Prompt
# =====================================

prompt = ChatPromptTemplate.from_template(
"""
Answer ONLY from the given context.

Context:
{context}

Question:
{question}
"""
)

# =====================================
# Chain
# =====================================

chain = prompt | llm

# =====================================
# Chat Loop
# =====================================

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    # Retrieve Documents
    docs = retriever.invoke(question)

    # Context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    print("\nAI : ", end="", flush=True)

    # Streaming Response
    for chunk in chain.stream(
        {
            "context": context,
            "question": question
        }
    ):

        print(
            chunk.content,
            end="",
            flush=True
        )

    print()
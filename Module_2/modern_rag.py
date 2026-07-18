from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------
# Embedding Model
# -----------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -----------------------------------
# Load FAISS Database
# -----------------------------------

vector_db = FAISS.load_local(
    "Module_2/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------------
# Retriever
# -----------------------------------

retriever = vector_db.as_retriever(
    search_kwargs={
        "k":3
    }
)

# -----------------------------------
# LLM
# -----------------------------------

llm = ChatOllama(
    model="llama3.2"
)

# -----------------------------------
# Prompt
# -----------------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
reply with:

"I don't know."

Context:
{context}

Question:
{question}
"""
)

# -----------------------------------
# Chain
# -----------------------------------

chain = prompt | llm | StrOutputParser()

# -----------------------------------
# Chat Loop
# -----------------------------------

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    # Retrieve Documents
    documents = retriever.invoke(question)

    # Convert Documents -> Context
    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # Generate Answer
    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    print("\nAI :")
    print(answer)
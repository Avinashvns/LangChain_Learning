from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Embedding Model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load Existing FAISS
vector_db = FAISS.load_local(
    "Module_2/multiple_vector_store", embeddings, allow_dangerous_deserialization=True
)

print("Vector Database Loaded Sucessfully")

# Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOllama(model="llama3.2")

# prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are an AI Assistant.

    Answer ONLY from the given context.

    If the answer is not found in the context,
    reply:

    I don't know.

    Context:
    {context}

    Question:
    {question}
"""
)

# chain
chain = prompt | llm | StrOutputParser()

# Chat Loop

while True:

    question = input("\nYou :")

    if question.lower() == "exit":
        break
    
    # Retrieve Relevant Documents
    docs = retriever.invoke(question)
    
    # Show Retriever Sources
    print("\nRetrieved Documents")
    print("-" *50)

    for i , doc in enumerate(docs, start=1):
        print(f"{i}. {doc.metadata['source']}")
    
    print("+" * 50)

    # Build Context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate Answer
    answer = chain.invoke(
        {
            "context" : context,
            "question" : question
        }
    )

    print("\nAI :")
    print(answer)

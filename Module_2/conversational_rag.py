from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# embedding Model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load FAISS
vector_db = FAISS.load_local(
    "Module_2/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever
retriever = vector_db.as_retriever(
    search_kwargs= { "k" : 3 }
)

# LLM
llm = ChatOllama(model="llama3.2")

# prompt
prompts = ChatPromptTemplate.from_messages(
    [
        ("system" , """
    Answer only from the given context.
         If answer is not available, 
         say "I don`t know."
         Context:
         {context}
"""),
MessagesPlaceholder("history"),
("human","{question}")
    ]
)

# chain
chain = prompts | llm | StrOutputParser()

# session Store
store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Runnable History

chatbot = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# chat
while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    answer = chatbot.invoke(
        {
            "question" : question,
            "context" : context
        },
        config ={
            "configurable" : {
                "session_id" : "user1"
            }
        }
    )

    print("\nAI :")
    print(answer)
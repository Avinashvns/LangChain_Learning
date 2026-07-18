from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# llm
llm = ChatOllama(model="llama3.2")

# prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a helpful AI Assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human" , "{input}")
    ]
)

# chain
chain = prompt | llm

# Memory Store
store = {}

def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Runnable with History
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Chat loop

while True:
    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    response = chatbot.invoke(
        {"input" : question},
        config= {
            "configurable" : {
                "session_id" : "user_1"
            }
        }
    )

    print("\nAI :", response.content)
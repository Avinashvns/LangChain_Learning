from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are a helpful AI assistant."),
    MessagesPlaceholder("history"),
    ("human" , "{question}")
])

chain = prompt | llm

# Memory
history = []

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    response = chain.invoke({
        "history" : history,
        "question" : question
    })

    print("AI : ", response.content)

    # Save User Message
    history.append(HumanMessage(content=question))

    # Save AI Message
    history.append(AIMessage(content=response.content))
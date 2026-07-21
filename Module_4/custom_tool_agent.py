from datetime import datetime
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Custom Tool
@tool
def current_time(dummy: str = "") -> str:
    """
    Return the current system time.
    """

    return datetime.now().strftime("%H:%M:%S")

# LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

llm = llm.bind_tools(
    [current_time]
)

# Chat
while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    response = llm.invoke(question)

    if response.tool_calls:

        tool_call = response.tool_calls[0]

        result = current_time.invoke(
            tool_call["args"]
        )

        print("\nCurrent Time :")
        print(result)

    else:

        print(response.content)
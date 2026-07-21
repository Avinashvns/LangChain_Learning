from datetime import datetime

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# Tool 1

@tool
def current_time(dummy: str = "") -> str:
    """
    Return current system time.
    """
    return datetime.now().strftime("%H:%M:%S")


# Tool 2

@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers.
    """
    return a + b


# LLM

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

llm_with_tools = llm.bind_tools(
    [current_time, add]
)

# Chat

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    response = llm_with_tools.invoke(question)

    # No Tool Needed
    if not response.tool_calls:

        print("\nAI :")
        print(response.content)
        continue

    tool_call = response.tool_calls[0]

    tool_name = tool_call["name"]

    if tool_name == "current_time":

        tool_result = current_time.invoke(
            tool_call["args"]
        )

    elif tool_name == "add":

        tool_result = add.invoke(
            tool_call["args"]
        )

    else:

        tool_result = "Unknown Tool"

    # Give Tool Result Back To LLM

    final_answer = llm.invoke(
        f"""
User Question:

{question}

Tool Result:

{tool_result}

Answer the user naturally.
"""
    )

    print("\nAI :")
    print(final_answer.content)
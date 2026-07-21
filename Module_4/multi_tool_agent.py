from datetime import datetime

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Tool 1
@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers.
    """
    return a + b

# Tool 2
@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b

# Tool 3
@tool
def current_time(dummy: str = "") -> str:
    """
    Return current system time.
    """
    return datetime.now().strftime("%H:%M:%S")

# Tool 4
@tool
def reverse_text(text: str) -> str:
    """
    Reverse the given text.
    """
    return text[::-1]

# Tool 5
@tool
def word_count(text: str) -> int:
    """
    Count total words.
    """
    return len(text.split())

# LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

llm = llm.bind_tools([
    add,
    multiply,
    current_time,
    reverse_text,
    word_count
])

# Chat Loop
while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    response = llm.invoke(question)

    if not response.tool_calls:

        print("\nAI :")
        print(response.content)
        continue

    tool_call = response.tool_calls[0]

    tool_name = tool_call["name"]

    print("\nSelected Tool :", tool_name)

    if tool_name == "add":
        result = add.invoke(tool_call["args"])

    elif tool_name == "multiply":
        result = multiply.invoke(tool_call["args"])

    elif tool_name == "current_time":
        result = current_time.invoke(tool_call["args"])

    elif tool_name == "reverse_text":
        result = reverse_text.invoke(tool_call["args"])

    elif tool_name == "word_count":
        result = word_count.invoke(tool_call["args"])

    else:
        result = "Unknown Tool"

    print("\nTool Result :")
    print(result)
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# Python Execution Tool

@tool
def execute_python(code: str) -> str:
    """
    Execute Python code and return local variables.
    """

    try:

        local_vars = {}

        exec(code, {}, local_vars)

        return str(local_vars)

    except Exception as e:

        return str(e)

# LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

llm = llm.bind_tools(
    [execute_python]
)

# Chat
while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    response = llm.invoke(question)

    # Tool required?
    if response.tool_calls:

        tool_call = response.tool_calls[0]

        print("\nTool Selected :")
        print(tool_call["name"])

        print("\nGenerated Python Code :")
        print(tool_call["args"]["code"])

        result = execute_python.invoke(
            tool_call["args"]
        )

        print("\nExecution Result :")
        print(result)

    else:

        print("\nAI :")
        print(response.content)
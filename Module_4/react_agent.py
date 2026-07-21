from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

# Tool 1
@tool
def add(a: int, b:int)-> int:
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

llm = ChatOllama(model="llama3.2",temperature=0)

# Agent

agent = create_agent(
    model = llm,
    tools = [add, multiply],
    system_prompt = """
        You are a helpful AI assistant.

        Whenever mathematical calculations are required,
        always use the available tools.
    """
)

# chat Loop

while True:
    question = input("\nYou : ")
    if question.lower() == "exit":
        break

    response = agent.invoke(
        {
            "messages" : [
                {
                    "role" : "user",
                    "content" : question
                }
            ]
        }
    )

    print("\nAI : ")
    print(response["messages"][-1].content)

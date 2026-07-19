from langchain_core.tools import tool

@tool
def add(a: int, b: int)-> int:
    """
        Add two numbers.
    """
    return a + b

@tool
def multiply(a: int, b:int)-> int:
    """
    Multiply two numbers.
    """
    return a * b

# String Tool
@tool
def uppercase(text: str) -> str:
    """
    Convert text to uppercase.
    """
    return text.upper()

print(add.invoke(
    {
        "a" : 10,
        "b" : 20
    }
))

print(multiply.invoke(
    {
        "a" :8,
        "b" : 5
    }
))

print(
    uppercase.invoke(
        {
            "text": "langchain"
        }
    )
)
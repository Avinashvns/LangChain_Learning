from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

def greet(name):
    return f"Hello {name}"

def square(number):
    return number * number

def uppercase(text):
    return text.upper()

chain = RunnableLambda(greet)
chain_n = RunnableLambda(square)
chain_text = RunnableLambda(uppercase)

result = chain.invoke("Akash")
result_n = chain_n.invoke(5)
result_text = chain_text.invoke("langChain")

print(result)
print(result_n)
print(result_text)


# Runnable Lambda + LLM
llm = ChatOllama(model="llama3.2")

def clean_text(text):
    return text.strip()

chain_llm = ( RunnableLambda(clean_text) | llm )

response = chain_llm.invoke("   What is Python?   ")

print(response.content)


# Multiple Runnable Lambda

def lower(text):

    return text.lower()


def remove_space(text):

    return text.strip()


chain_multi = (

    RunnableLambda(remove_space)

    |

    RunnableLambda(lower)

)

print(

    chain_multi.invoke(

        "   HELLO LANGCHAIN   "

    )

)
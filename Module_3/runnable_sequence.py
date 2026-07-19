from langchain_core.runnables import RunnableLambda

step1 = RunnableLambda(
    lambda x: x.upper()
)

step2 = RunnableLambda(
    lambda x: x[::-1]
)

chain = step1 | step2

print(
    chain.invoke("langchain")
)
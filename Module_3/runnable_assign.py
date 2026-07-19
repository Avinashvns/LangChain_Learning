from langchain_core.runnables import RunnablePassthrough , RunnableLambda

chain = RunnablePassthrough.assign(
    length = RunnableLambda(
        lambda x: len(x["text"])
    )
)

result = chain.invoke({
        "text": "LangChain"
    })
print(result)

# Multiple Fields

chains = RunnablePassthrough.assign(
    upper = lambda x: x["text"].upper(),
    lower = lambda x: x["text"].lower()
)

results = chains.invoke({
    "text" : "Langchain"
})
print(results)


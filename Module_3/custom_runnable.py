from langchain_core.runnables import Runnable

class UpperCaseRunnable(Runnable):

    def invoke(self, input, config=None):
        return input.upper()

chain = UpperCaseRunnable()

print(chain.invoke("langChain"))
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Module_1/python.pdf")

documents = loader.load()

print(documents[0].metadata["page"])
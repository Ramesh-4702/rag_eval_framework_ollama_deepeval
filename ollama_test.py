from langchain_ollama import OllamaLLM

llm =OllamaLLM(model="llama3.2:latest")

response = llm.invoke("Hello! Confirm you are running locally on Ollama.") 
print("Ollama Response:", response)
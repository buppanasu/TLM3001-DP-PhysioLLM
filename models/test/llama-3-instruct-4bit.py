from langchain_community.llms import Ollama
import time

llm = Ollama(model="llama3")

query = "Tell me a joke, make it long and boring"

start_time = time.time()
token_count = 0

for chunks in llm.stream(query):
    token_count += 1
    if time.time() - start_time > 1:
        print("\nTokens in one 1 sec: ", token_count)
        token_count = 0
        start_time = time.time()
    print(chunks, end="")

from dotenv import load_dotenv
from core.my_llm import MyLLM

load_dotenv()

llm = MyLLM(provider="modelscope")

messages = [{"role":"user", "content":"Hi! Tell me about yourself"}]

response_stream = llm.think(messages)

print(response_stream)

for chunk in response_stream:
    pass
from dotenv import load_dotenv

from helloagent.core.hello_agents import HelloAgentsLLM


def main() -> None:
    load_dotenv()
    llm = HelloAgentsLLM()
    messages = [{"role": "user", "content": "Hi! Tell me about yourself"}]

    for chunk in llm.stream_invoke(messages):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()

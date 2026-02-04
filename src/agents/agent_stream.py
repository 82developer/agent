import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


def main() -> None:      
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    model = ChatDeepSeek(
            model= "deepseek-chat",
            temperature=0
            )
    full = None
    for chunk in model.stream("what is the weather in sf"):
        full = chunk if full is None else full + chunk
        print(full.text)
    print("-------------")
    print(full.content_blocks)


if __name__ == "__main__":
    main()

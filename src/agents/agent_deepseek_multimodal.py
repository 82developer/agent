import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

def main() -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    model = ChatDeepSeek(
            model="deepseek-chat", 
            temperature=0
            )
    response = model.invoke("Create a picture of a cat")
    print(response.content_blocks)


if __name__ == "__main__":
    main()

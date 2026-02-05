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

    tool = {"type":"web search"}
    model_with_tool = model.bind_tools(tool)

    response = model_with_tool.invoke("What was a positive news story from today?")
    print(response.content_blocks)

if __name__ == "__main__":
    main()

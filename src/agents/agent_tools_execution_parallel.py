import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    model = ChatDeepSeek(model="deepseek-chat", temperature=0)

    model_with_tools = model.bind_tools([get_weather])

    response = model_with_tools.invoke("What's the weather in Boston and Tokyo?")
    print(response.tool_calls)

    results = []
    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_weather":
            result = get_weather.invoke(tool_call)
        results.append(result)
    print(results)


if __name__ == "__main__":
    main()

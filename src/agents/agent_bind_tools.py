import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

def main() -> None:      
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    model = ChatDeepSeek(
            model= "deepseek-chat",
            temperature=0
            )
    model_with_tools = model.bind_tools([get_weather])

    response = model_with_tools.invoke("What's the weather like in Boston?")
    print(response.text)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool: {tool_call['name']}")
            print(f"Args: {tool_call['args']}")
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            if tool_name == "get_weather":
                result = get_weather.invoke(tool_args)
                print(result)



if __name__ == "__main__":
    main()

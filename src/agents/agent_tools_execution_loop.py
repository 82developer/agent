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
    model_with_tools= model.bind_tools([get_weather])

    messages = [{"role":"user","content":"What's the weather like in Boston?"}]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        tool_result = get_weather.invoke(tool_call)
        messages.append(tool_result)

    final_response = model_with_tools.invoke(messages)
    print(final_response.text)



if __name__ == "__main__":
    main()

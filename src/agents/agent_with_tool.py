import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage, AIMessage, ToolMessage

def pretty_print_messages(messages):
    for i, msg in enumerate(messages, start=1):
        if isinstance(msg, HumanMessage):
            print(f"\n[{i}] 👤 USER")
            print(msg.content)

        elif isinstance(msg, ToolMessage):
            print(f"\n[{i}] 🛠️ TOOL: {msg.name}")
            print(msg.content)

        elif isinstance(msg, AIMessage):
            print(f"\n[{i}] 🤖 ASSISTANT")
            print(msg.content)

            # If the AI requested tool calls, show them in a clean way
            if getattr(msg, "tool_calls", None):
                print("   └─ tool_calls:")
                for call in msg.tool_calls:
                    print(f"      • {call['name']}({call['args']})  id={call['id']}")
@tool 
def search(
           query: str
          )-> str:
    """Search for information."""
    result = f"Results for: {query}"
    print(1)
    return result

@tool
def get_weather(
        location: str
        ) -> str:
    """Get weather information for a location."""
    result = f"Weather in {location}: Sunny, 72 F"
    print(2)
    return result

def main()->None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    llm = ChatDeepSeek(
            model= "deepseek-chat",
            temperature=0
            )
    agent = create_agent(
            model=llm,
            tools=[search, get_weather]
            )
    result = agent.invoke({"messages": [{"role":"user","content":"what is the weather in sf"}]})
    pretty_print_messages(result["messages"])


if __name__ == "__main__": main()

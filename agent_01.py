import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage, AIMessage, ToolMessage

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in  {city}"

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

def main() -> None:
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
            tools=[get_weather],
            system_prompt="You are a helpful assistant"
            )

    result = agent.invoke({"messages": [{"role":"user","content":"what is the weather in sf"}]})
    pretty_print_messages(result["messages"])

if __name__ == "__main__":
    main()

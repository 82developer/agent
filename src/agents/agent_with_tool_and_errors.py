import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

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

@wrap_tool_call
def handle_tool_errors(request, handler):
    """handler tool execution errors with custom messages"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
                context=f"Tool error: Please check you input and try again:({str(e)})",
                tool_call_id=request.tool_call["id"]
                )


@tool 
def search(
           query: str
          )-> str:
    """Search for information."""
    raise RuntimeError("error 1")
    return f"Results for: {query}"

@tool
def get_weather(
        location: str
        ) -> str:
    """Get weather information for a location."""
    raise RuntimeError("Error 2")
    return f"Weather in {location}: Sunny, 72 F"

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
            tools=[search, get_weather],
            middleware=[handle_tool_errors]
            )
    result = agent.invoke({"messages": [{"role":"user","content":"what is the weather in sf"}]})
    pretty_print_messages(result["messages"])



if __name__ == "__main__": main()

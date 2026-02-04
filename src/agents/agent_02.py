from dataclasses import dataclass
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in  {city}"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: str | None = None

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

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

def main() -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    llm = ChatDeepSeek(
            model= "deepseek-chat",
            temperature=0
            )

    checkpointer = InMemorySaver()

    agent = create_agent(
            model = llm,
            system_prompt = SYSTEM_PROMPT,
            tools = [get_weather_for_location, get_user_location],
            context_schema= Context,
            response_format=ToolStrategy(ResponseFormat),
            checkpointer=checkpointer
            )
    config = {"configurable":{"thread_id":1}}

    response = agent.invoke(
            {"messages":[
                {"role":"user","content":'what is the weather outside?'}
                ]},
            config=config,
            context=Context(user_id="1")
            )
    pretty_print_messages(response["messages"])
    print(response['structured_response'])

    response = agent.invoke(
            {"messages":[{"role":"user","content":"thank you!"}]},
            config=config,
            context=Context(user_id="1")
            )
    pretty_print_messages(response["messages"])
    print(response['structured_response'])


if __name__ == "__main__":
    main()

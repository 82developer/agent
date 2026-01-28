from dataclasses import dataclass
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek


@dataclass
class Context:
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role"""
    user_role = getattr(request.runtime.context, "user_role", None)
    base_prompt = "You are a helpful assistant"

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt


@tool
def web_search() -> str:
    """search the web"""
    return "search result"


def main() -> None:
    load_dotenv()
    model = ChatDeepSeek(model="deepseek-chat", temperature=0)
    agent = create_agent(
        model=model,
        tools=[web_search],
        middleware=[user_role_prompt],
        context_schema=Context,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Explain machine learning"}]},
        context={"user_role": "expert"},
    )
    print(result)


if __name__ == "__main__":
    main()

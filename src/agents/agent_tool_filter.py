from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (ModelRequest, ModelResponse,
                                         wrap_model_call)
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek


@dataclass
class UserContext:
    """Custom runtime context schema."""
    user_role: str

@tool
def read_data() -> str:
    """Read data from the system."""
    return "Data read successfully"

@tool
def write_data(data: str) -> str:
    """Write data to the system."""
    return f"Data writtem: {data}"

@tool
def delete(record_id: int) -> str:
    """Delete data from the system."""
    return f"Record {record_id} deleted successfully"

@wrap_model_call
def filter_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
    """Filter tools based on user permissions."""
    user_role = getattr(request.runtime.context,"user_role",None)

    if user_role == "admin":
        tools = request.tools
    else:
        tools = [t for t in request.tools if getattr(t,'name').startswith("read_")]
    
    return handler(request.override(tools=tools))

load_dotenv()
llm_deep_seek = ChatDeepSeek(
                model= "deepseek-chat",
                temperature=0
                )
agent = create_agent(
        model=llm_deep_seek,
        tools=[read_data,write_data,delete],
        middleware=[filter_tools],
        context_schema=UserContext
        )
result = agent.invoke(
                        {"messages": [
                            {"role":"user","content":"what is the weather in sf"}
                            ]
                        },
                        context={"user_role": "admin"}
                    )
print(result)

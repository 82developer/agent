from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek

class ContacInfor(BaseModel):
    name: str
    email: str
    phone: str

@tool
def search_tool(query: str) -> str:
    """return query"""
    return query

def main() -> None:
    load_dotenv()
    model = ChatDeepSeek(model="deepseek-chat", temperature=0)
    agent = create_agent(
        model=model,
        tools=[search_tool],
        response_format=ToolStrategy(ContacInfor)
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567",
                }
            ]
        }
    )

    print(result["structured_response"])

if __name__ == "__main__":
    main()    

from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

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
            tools=[]
            )

    # ---- Memory = message history ----
    chat_history: list = []

    # Turn 1
    chat_history.append(HumanMessage(content="My name is Alvaro"))
    r1 = agent.invoke({"messages": chat_history})
    chat_history.append(AIMessage(content=r1["messages"][-1].content))
    print(r1["messages"][-1].content)

    # Turn 2
    chat_history.append(HumanMessage(content="What is my name?"))
    r2 = agent.invoke({"messages": chat_history})
    print(r2["messages"][-1].content)

    final_answer = r2["messages"][-1].content
    chat_history.append(AIMessage(content=final_answer))

    print("\nAgent response:")
    print(final_answer)



if __name__ == "__main__":
    main()

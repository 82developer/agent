from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage

def main() -> None:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("toke not found")

    llm = ChatGoogleGenerativeAI(
            model = "models/gemini-2.5-flash",
            temperature = 0
            )

    chat_history: list = []

    agent = create_agent(
            model= llm,
            tools =[]
            )

    chat_history.append(HumanMessage(content="My name is Alvaro"))
    result1 = agent.invoke({"messages":chat_history})
    print(result1)

    chat_history.append(HumanMessage(content="What is my name?"))
    result2 = agent.invoke({"messages":chat_history})

    print(result2)


if __name__ == "__main__":
    main()

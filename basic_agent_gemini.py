"""
Minimal LangChain Agent using Gemini (Python 3).
Purpose: Verify LangChain + Gemini integration works.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent


# ---- Tool (minimal) ----
@tool
def multiply(a: int, b: int) -> int:
    """Return the multiplication of two integers."""
    return a * b


def main() -> None:
    # Load environment variables
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found in environment variables.")

    # ---- LLM (Gemini) ----
    llm = ChatGoogleGenerativeAI(
        # ⚠️ USE EXACT MODEL NAME FROM ListModels
        # Example (replace with the one that worked for you):
        model="models/gemini-2.5-flash",
        temperature=0.0,
    )

    # ---- Create agent ----
    tools = [multiply]
    agent = create_agent(
            llm,
            tools)

    # ---- Run agent ----
    question = "Calculate 6 multiplied by 7 using the tool."

    result = agent.invoke(
        {"messages": [("user", question)]}
    )

    # ---- Print final answer ----
    print("\nAgent final answer:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

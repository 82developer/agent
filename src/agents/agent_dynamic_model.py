import os
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from langchain.agents import create_agent

load_dotenv()
llm_gemini = ChatGoogleGenerativeAI(
             model = "models/gemini-2.5-flash",
             temperature = 0
             )

llm_deep_seek = ChatDeepSeek(
                model= "deepseek-chat",
                temperature=0
                )

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity"""
    message_count = len(request.state["messages"])
    if message_count > 10:
       print('1')
       model = llm_deep_seek
    else: 
       print('2')
       model = llm_gemini

    return handler(request.override(model=model))

def main() -> None:

    api_key_gemini =  os.getenv("GOOGLE_API_KEY")
    if not api_key_gemini:
        raise RuntimeError("GOOGLE_API_KEY not found in environment variables.")

    api_key_deepseek =  os.getenv("DEEPSEEK_API_KEY")
    if not api_key_deepseek:
        raise RuntimeError("DEEPSEEK_API_KEY not found")

    agent = create_agent(
            model=llm_deep_seek,
            tools=[],
            middleware=[dynamic_model_selection]
            )

    question = "Calculate 6 multiplied by 7 using the tool."
    result = agent.invoke(
        {"messages": [("user", question)]}
    )

    print(result["messages"][-1].content)



if __name__ == "__main__":
    main()

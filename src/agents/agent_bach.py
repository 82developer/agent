import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


def main() -> None:      
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    model = ChatDeepSeek(
            model= "deepseek-chat",
            temperature=0
            )
    responses = model.batch([
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is queantum computing?"
        ])
    for response in responses:
        print("-------")
        print(response)



if __name__ == "__main__":
    main()

import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    model = ChatDeepSeek(
            model="deepseek-chat", 
            temperature=0
            )

    response = model.invoke("Why do parrots have colorful feathers?")
    reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
    #print(reasoning_steps if reasoning_steps else chunk.text)
    print(" ".join(step["reasoning"] for step in reasoning_steps))

    print("fin")



if __name__ == "__main__":
    main()

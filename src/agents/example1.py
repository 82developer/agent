import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def main() -> None:
    # Load .env if present
    #load_dotenv()

    # 1. Verify token exists
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment variables.")

    # 2. Create Gemini client
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # very stable for tests
        temperature=0.0
    )

    # 3. Make the smallest possible request
    response = llm.invoke("Reply ONLY with the word: CONNECTED")

    # 4. Print result
    print("Gemini response:")
    print(response.content)


if __name__ == "__main__":
    main()

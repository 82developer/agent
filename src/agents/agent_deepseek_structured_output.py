import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

class Movie (BaseModel):
    """A Movie with details."""
    title: str = Field(...,description="The title of the movie")
    year: int = Field(...,description = "The year the movie was released")
    director: str = Field(...,description="The director of the movie")
    rating: float = Field(...,description="The movie's rating out of 10")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("token not found")

    model = ChatDeepSeek(
            model="deepseek-chat", 
            temperature=0
            )

    model_with_tools = model.with_structured_output(Movie,include_raw=True)
    response = model_with_tools.invoke("Provide details about the movie Inception")
    print(response)

if __name__ == "__main__":
    main()

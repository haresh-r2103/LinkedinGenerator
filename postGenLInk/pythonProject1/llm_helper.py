from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load environment variables from .env file
load_dotenv()



# Initialize the LLM
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-vision-preview")

if __name__ == "__main__":
    response = llm.invoke("Tell me the recipe for Burger")
    print(response.content)

from langchain_community.chat_models import ChatOpenAI
from config.config import Config

api_key = Config.OPENAI_API_KEY
print(f"Using API Key: {api_key[:4]}... (hidden for security)")  # Partial print for security

def load_llm():
    """Load OpenAI LLM model."""
    return ChatOpenAI(model_name="gpt-4",temperature=Config.TEMPERATURE)

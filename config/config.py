import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

class Config:
    """Configuration class for storing API keys and settings."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TEMPERATURE = os.getenv("TEMPERATURE")

    @staticmethod
    def check_keys():
        """Ensure API keys are loaded correctly."""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing. Set it in the .env file.")


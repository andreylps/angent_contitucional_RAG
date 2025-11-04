# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    chroma_db_path: str = "./chroma_db"
    model_name: str = "gpt-4o-mini"
    
    class Config:
        env_file = ".env"

settings = Settings()
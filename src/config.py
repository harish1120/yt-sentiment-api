from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HF_TOKEN: str

    class Config: 
        env_file = "/Users/harishsundaralingam/myworkspace/sentiment_analysis/.env"


settings = Settings()
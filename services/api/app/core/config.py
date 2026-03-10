from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore[import-not-found]


class Settings(BaseSettings):
    app_name: str = "AI Text Origin Risk Analyzer API"
    api_v1_prefix: str = "/v1"
    api_env: str = "development"
    frontend_origin: str = "http://localhost:3000"
    max_upload_size_mb: int = 20
    retention_hours: int = 24
    batch_limit: int = 10
    default_language: str = "en"
    min_tokens_for_medium_confidence: int = 120
    min_tokens_for_high_confidence: int = 300
    enable_artifact_models: bool = False
    ml_artifact_dir: str | None = None
    artifact_device: str = "auto"

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

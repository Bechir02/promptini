"""Typed application configuration.

Reads from environment variables only (the app's existing ``load_dotenv()`` call
populates the environment at runtime; this module never opens a secrets file
itself). Provides fail-fast validation so a missing key produces a clear message
instead of a deep stack trace at the first API call.

Providers are fixed to Groq (primary) and Cerebras (fallback) by design.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pull only from the process environment. We deliberately do NOT register an
    # env_file here so this module never reads a .env directly.
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # ── Provider keys (Groq primary, Cerebras fallback) ──────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    github_token: str = Field(default="", alias="GITHUB_TOKEN")

    # ── Optional extra providers (OpenAI-compatible; used only if key set) ────
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    together_api_key: str = Field(default="", alias="TOGETHER_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openrouter_model: str = Field(default="meta-llama/llama-3.3-70b-instruct", alias="OPENROUTER_MODEL")
    together_model: str = Field(default="meta-llama/Llama-3.3-70B-Instruct-Turbo", alias="TOGETHER_MODEL")
    google_model: str = Field(default="gemini-1.5-flash", alias="GOOGLE_MODEL")

    # ── Model names ──────────────────────────────────────────────────────────
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    cerebras_model: str = Field(default="llama3.1-8b", alias="CEREBRAS_MODEL")

    # ── LLM call params ──────────────────────────────────────────────────────
    max_tokens: int = Field(default=1500, alias="PF_MAX_TOKENS")
    temperature: float = Field(default=0.4, alias="PF_TEMPERATURE")

    # ── Rate limiting (per client, sliding window; <=0 disables) ─────────────
    rate_limit_calls: int = Field(default=20, alias="PF_RATE_LIMIT_CALLS")
    rate_limit_window: float = Field(default=60.0, alias="PF_RATE_LIMIT_WINDOW")

    # ── Paths ────────────────────────────────────────────────────────────────
    prompts_file: str = Field(default="prompts.json", alias="PF_PROMPTS_FILE")
    db_path: str = Field(default="lancedb_store", alias="PF_DB_PATH")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="PF_EMBEDDING_MODEL"
    )

    @property
    def has_any_provider(self) -> bool:
        return bool(
            self.groq_api_key or self.cerebras_api_key or self.openrouter_api_key
            or self.together_api_key or self.google_api_key
        )

    def require_provider(self) -> None:
        """Raise a clear error at startup if no provider key is configured."""
        if not self.has_any_provider:
            raise RuntimeError(
                "No LLM provider configured. Set GROQ_API_KEY (primary) and/or "
                "CEREBRAS_API_KEY (fallback) in your environment / Space secrets."
            )


# Singleton-style accessor so callers share one parsed instance.
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

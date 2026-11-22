"""
Configuration properties for Weather Information Search MCP Server.

Self-contained module configuration designed to support standalone deployment,
externalized endpoints, and zero-coupling with host application settings.
Exclusively manages properties for the 3 implemented MCP tools:
  1. get_current_weather (/current.json)
  2. get_weather_forecast (/forecast.json)
  3. get_air_quality (aqi=yes on /current.json)
"""

from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WeatherConfig(BaseSettings):
    """
    Externalized configuration properties for WeatherAPI.com integration.
    
    Can be configured via environment variables, .env file, or direct instantiation.
    Supports running within the main application or in an isolated standalone repository.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    # Authentication & API Credentials
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("WEATHERAPI_KEY", "WEATHER_API_KEY"),
        description="API key for WeatherAPI.com",
    )

    # Base URL & Service Host
    base_url: str = Field(
        default="https://api.weatherapi.com/v1",
        validation_alias=AliasChoices("WEATHERAPI_BASE_URL", "WEATHER_BASE_URL"),
        description="Base URL for WeatherAPI service",
    )

    # Endpoints for implemented tools
    current_endpoint: str = Field(
        default="/current.json",
        validation_alias=AliasChoices("WEATHERAPI_CURRENT_ENDPOINT", "WEATHER_CURRENT_ENDPOINT"),
        description="Endpoint path for real-time current weather metrics and air quality",
    )
    forecast_endpoint: str = Field(
        default="/forecast.json",
        validation_alias=AliasChoices("WEATHERAPI_FORECAST_ENDPOINT", "WEATHER_FORECAST_ENDPOINT"),
        description="Endpoint path for multi-day weather forecasts",
    )

    # Execution & Telemetry Properties
    timeout_seconds: float = Field(
        default=10.0,
        validation_alias=AliasChoices("WEATHERAPI_TIMEOUT_SECONDS", "WEATHER_TIMEOUT_SECONDS"),
        description="HTTP client timeout in seconds",
    )
    aqi_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("WEATHERAPI_AQI_ENABLED", "WEATHER_AQI_ENABLED"),
        description="Whether to request Air Quality Index data with weather calls",
    )

    # ── Derived Full URLs ───────────────────────────────────────────────────

    @property
    def current_url(self) -> str:
        """Constructs full URL for the current weather & air quality endpoint."""
        return f"{self.base_url.rstrip('/')}/{self.current_endpoint.lstrip('/')}"

    @property
    def forecast_url(self) -> str:
        """Constructs full URL for the forecast endpoint."""
        return f"{self.base_url.rstrip('/')}/{self.forecast_endpoint.lstrip('/')}"


_config_instance: Optional[WeatherConfig] = None


def get_weather_config() -> WeatherConfig:
    """Singleton getter for WeatherConfig."""
    global _config_instance
    if _config_instance is None:
        _config_instance = WeatherConfig()
    return _config_instance

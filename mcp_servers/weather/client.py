"""
Asynchronous Weather Client for WeatherAPI.com.

Strictly integrates with https://api.weatherapi.com without external fallback.
All URLs, endpoints, and timeouts are externalized into properties via WeatherConfig.
Exhaustively handles all HTTP response codes (200, 400, 401, 403, 404, 429, 5xx)
and error schemas (error400, error401, error403) defined in the WeatherAPI Swagger spec.
This module is self-contained within mcp_servers to enable independent deployment.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Ensure package paths are present on sys.path for standalone or subshell execution
_mcp_servers_root = Path(__file__).resolve().parent.parent
_project_root = _mcp_servers_root.parent
for _p in (_project_root, _mcp_servers_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import httpx
from loguru import logger

from mcp_servers.weather.config import WeatherConfig, get_weather_config


class WeatherClient:
    """
    Fetches real-time weather, multi-day forecasts, and air quality metrics
    strictly from WeatherAPI.com using externalized configuration properties.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[WeatherConfig] = None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = get_weather_config()

        # Allow explicit parameter overrides
        if api_key is not None:
            self.config = self.config.model_copy(update={"api_key": api_key.strip()})
        if base_url is not None:
            self.config = self.config.model_copy(update={"base_url": base_url.strip()})

    @property
    def api_key(self) -> str:
        return self.config.api_key

    # ── Swagger-Aligned HTTP Error Handler ──────────────────────────────────

    @staticmethod
    def _handle_http_error(resp: httpx.Response, city: str) -> dict[str, Any]:
        """
        Maps HTTP status codes and error payloads as defined in the WeatherAPI SwaggerHub 1.0.2 spec:
          • HTTP 400: error400 — Invalid parameters (q missing: 1003, invalid url: 1005, location not found: 1006)
          • HTTP 401: error401 — Authentication failure (key missing: 1002, key invalid: 2006)
          • HTTP 403: error403 — Authorization failure (quota exceeded: 2007, disabled: 2008, forbidden resource: 2009)
          • HTTP 404: Endpoint path not found
          • HTTP 429: Too Many Requests / Rate limit exceeded
          • HTTP 5xx: Upstream service failure (internal error: 9999)
        """
        status_code = resp.status_code
        error_payload = {}
        try:
            error_payload = resp.json().get("error", {})
        except Exception:
            pass

        code = error_payload.get("code")
        raw_msg = error_payload.get("message", resp.reason_phrase or "Unknown error")

        if status_code == 400:
            category = "BAD_REQUEST"
            if code == 1006:
                category = "LOCATION_NOT_FOUND"
                hint = f"Location '{city}' was not found. Please verify spelling or pass latitude,longitude."
            elif code == 1003:
                category = "PARAMETER_MISSING"
                hint = "Parameter 'q' (location query) was not provided."
            elif code == 1005:
                category = "INVALID_REQUEST_URL"
                hint = "API request URL is invalid."
            else:
                hint = "The request parameters are invalid."

        elif status_code == 401:
            category = "UNAUTHORIZED"
            if code == 1002:
                hint = "API key was not provided in request."
            elif code == 2006:
                hint = "API key provided is invalid. Please check WEATHERAPI_KEY in your configuration."
            else:
                hint = "Authentication failed. Verify your WeatherAPI API key."

        elif status_code == 403:
            category = "FORBIDDEN"
            if code == 2007:
                category = "QUOTA_EXCEEDED"
                hint = "API key has exceeded calls per month quota."
            elif code == 2008:
                category = "ACCOUNT_DISABLED"
                hint = "API key has been disabled by WeatherAPI."
            elif code == 2009:
                category = "RESOURCE_FORBIDDEN"
                hint = "API key does not have access to the requested endpoint or resource for your plan."
            else:
                hint = "Access forbidden for this API key or subscription tier."

        elif status_code == 404:
            category = "NOT_FOUND"
            hint = f"WeatherAPI endpoint URL path was not found: {resp.url.path if hasattr(resp, 'url') else 'unknown'}"

        elif status_code == 429:
            category = "RATE_LIMITED"
            hint = "WeatherAPI rate limit exceeded. Please back off and retry."

        elif status_code >= 500:
            category = "SERVER_ERROR"
            hint = "WeatherAPI upstream service encountered an internal error. Please retry shortly."

        else:
            category = "HTTP_ERROR"
            hint = f"Unexpected HTTP status {status_code} received from WeatherAPI."

        logger.error(
            f"WeatherAPI [{status_code} {category}] for '{city}': {raw_msg} (code={code})"
        )

        return {
            "error": f"WeatherAPI error ({status_code} {category}): {raw_msg}",
            "error_category": category,
            "status_code": status_code,
            "code": code,
            "message": raw_msg,
            "hint": hint,
            "city": city,
            "provider": "WeatherAPI.com",
        }

    # ── Tool Methods ────────────────────────────────────────────────────────

    async def get_current_weather(self, city: str) -> dict[str, Any]:
        """
        Get current real-time atmospheric metrics for a city from WeatherAPI.com.

        Args:
            city: Name of the city (e.g. 'Wellington', 'London', 'Tokyo').
        """
        city_clean = city.strip()
        if not city_clean:
            return {
                "error": "City name must not be empty",
                "error_category": "PARAMETER_MISSING",
                "code": 1003,
                "provider": "WeatherAPI.com",
            }

        if not self.config.api_key:
            return {
                "error": "WeatherAPI API key not configured. Set WEATHERAPI_KEY in environment or properties.",
                "error_category": "UNAUTHORIZED",
                "status_code": 401,
                "code": 1002,
                "hint": "Set WEATHERAPI_KEY in .env or via WEATHERAPI_KEY environment variable.",
                "provider": "WeatherAPI.com",
                "city": city_clean,
            }

        url = self.config.current_url
        params = {
            "key": self.config.api_key,
            "q": city_clean,
            "aqi": "yes" if self.config.aqi_enabled else "no",
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                resp = await client.get(url, params=params)

                if resp.is_error:
                    return self._handle_http_error(resp, city_clean)

                data = resp.json()

            loc = data.get("location", {})
            curr = data.get("current", {})
            cond = curr.get("condition", {})
            aqi_raw = curr.get("air_quality", {})

            epa_index = aqi_raw.get("us-epa-index")
            epa_labels = {
                1: "Good (Minimal Impact)",
                2: "Moderate (Acceptable)",
                3: "Unhealthy for Sensitive Groups",
                4: "Unhealthy",
                5: "Very Unhealthy",
                6: "Hazardous",
            }

            return {
                "provider": "WeatherAPI.com",
                "city": loc.get("name", city_clean),
                "region": loc.get("region"),
                "country": loc.get("country"),
                "coordinates": {"lat": loc.get("lat"), "lon": loc.get("lon")},
                "temperature_c": curr.get("temp_c"),
                "feelslike_c": curr.get("feelslike_c"),
                "humidity_percent": curr.get("humidity"),
                "condition": cond.get("text", "Unknown"),
                "wind_kph": curr.get("wind_kph"),
                "wind_dir": curr.get("wind_dir"),
                "precipitation_mm": curr.get("precip_mm"),
                "pressure_mb": curr.get("pressure_mb"),
                "uv_index": curr.get("uv"),
                "air_quality": {
                    "us_epa_index": epa_index,
                    "air_quality_level": epa_labels.get(epa_index, "Moderate"),
                    "pm2_5_ug_m3": round(aqi_raw.get("pm2_5", 0), 1) if aqi_raw.get("pm2_5") is not None else None,
                    "pm10_ug_m3": round(aqi_raw.get("pm10", 0), 1) if aqi_raw.get("pm10") is not None else None,
                },
                "observed_at_localtime": loc.get("localtime"),
            }

        except httpx.TimeoutException:
            logger.error(f"WeatherAPI request timed out after {self.config.timeout_seconds}s for '{city_clean}'")
            return {
                "error": f"WeatherAPI request timed out after {self.config.timeout_seconds} seconds",
                "error_category": "TIMEOUT",
                "status_code": 408,
                "hint": "Check network connectivity or increase WEATHERAPI_TIMEOUT_SECONDS.",
                "provider": "WeatherAPI.com",
                "city": city_clean,
            }
        except httpx.ConnectError as e:
            logger.error(f"WeatherAPI connection failure for '{city_clean}': {e}")
            return {
                "error": f"WeatherAPI connection failure: {str(e)}",
                "error_category": "CONNECTION_ERROR",
                "status_code": 503,
                "hint": "Unable to connect to api.weatherapi.com. Check internet connection and DNS.",
                "provider": "WeatherAPI.com",
                "city": city_clean,
            }
        except Exception as e:
            logger.error(f"WeatherAPI unexpected failure for '{city_clean}': {e}")
            return {
                "error": f"WeatherAPI unexpected error: {str(e)}",
                "error_category": "UNEXPECTED_ERROR",
                "status_code": 500,
                "provider": "WeatherAPI.com",
                "city": city_clean,
            }

    async def get_weather_forecast(self, city: str, days: int = 3) -> dict[str, Any]:
        """
        Get multi-day forecast for a city from WeatherAPI.com.

        Args:
            city: Name of the city.
            days: Forecast duration in days (1 to 14 as per Swagger spec, default: 3).
        """
        city_clean = city.strip()
        if not city_clean:
            return {
                "error": "City name must not be empty",
                "error_category": "PARAMETER_MISSING",
                "code": 1003,
                "provider": "WeatherAPI.com",
            }

        if not self.config.api_key:
            return {
                "error": "WeatherAPI API key not configured. Set WEATHERAPI_KEY in environment or properties.",
                "error_category": "UNAUTHORIZED",
                "status_code": 401,
                "code": 1002,
                "hint": "Set WEATHERAPI_KEY in .env or via WEATHERAPI_KEY environment variable.",
                "provider": "WeatherAPI.com",
                "city": city_clean,
            }

        days = max(1, min(days, 14))
        url = self.config.forecast_url
        params = {
            "key": self.config.api_key,
            "q": city_clean,
            "days": days,
            "aqi": "yes" if self.config.aqi_enabled else "no",
            "alerts": "no",
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                resp = await client.get(url, params=params)

                if resp.is_error:
                    return self._handle_http_error(resp, city_clean)

                data = resp.json()

            loc = data.get("location", {})
            forecast_days = []
            for f in data.get("forecast", {}).get("forecastday", []):
                day_info = f.get("day", {})
                forecast_days.append({
                    "date": f.get("date"),
                    "max_temp_c": day_info.get("maxtemp_c"),
                    "min_temp_c": day_info.get("mintemp_c"),
                    "avg_temp_c": day_info.get("avgtemp_c"),
                    "chance_of_rain_percent": day_info.get("daily_chance_of_rain"),
                    "condition": day_info.get("condition", {}).get("text"),
                    "max_wind_kph": day_info.get("maxwind_kph"),
                    "uv_index": day_info.get("uv"),
                })

            return {
                "provider": "WeatherAPI.com",
                "city": loc.get("name", city_clean),
                "region": loc.get("region"),
                "country": loc.get("country"),
                "forecast_days": forecast_days,
            }

        except httpx.TimeoutException:
            return {
                "error": f"WeatherAPI request timed out after {self.config.timeout_seconds} seconds",
                "error_category": "TIMEOUT",
                "status_code": 408,
                "provider": "WeatherAPI.com",
                "city": city_clean,
            }
        except httpx.ConnectError as e:
            return {
                "error": f"WeatherAPI connection failure: {str(e)}",
                "error_category": "CONNECTION_ERROR",
                "status_code": 503,
                "provider": "WeatherAPI.com",
                "city": city_clean,
            }
        except Exception as e:
            return {
                "error": f"WeatherAPI unexpected error: {str(e)}",
                "error_category": "UNEXPECTED_ERROR",
                "status_code": 500,
                "provider": "WeatherAPI.com",
                "city": city_clean,
            }

    async def get_air_quality(self, city: str) -> dict[str, Any]:
        """
        Get Air Quality Index (AQI), PM2.5, PM10, and UV index from WeatherAPI.com.

        Args:
            city: Name of the city.
        """
        city_clean = city.strip()
        if not city_clean:
            return {
                "error": "City name must not be empty",
                "error_category": "PARAMETER_MISSING",
                "code": 1003,
                "provider": "WeatherAPI.com",
            }

        current_data = await self.get_current_weather(city_clean)
        if "error" in current_data:
            return current_data

        return {
            "provider": "WeatherAPI.com",
            "city": current_data.get("city", city_clean),
            "region": current_data.get("region"),
            "country": current_data.get("country"),
            "air_quality": current_data.get("air_quality", {}),
            "uv_index": current_data.get("uv_index"),
        }

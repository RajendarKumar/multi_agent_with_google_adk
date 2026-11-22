"""
Unit and Integration tests for Weather Information Search MCP Server.
Tests WeatherConfig, WeatherClient (pure WeatherAPI.com without fallback),
MCP Server tool schemas, tool execution, and ADK McpToolset integration.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch
import httpx

from mcp_servers.weather.config import WeatherConfig
from mcp_servers.weather.client import WeatherClient
from mcp_servers.weather.server import server as weather_mcp_server
from mcp_registry.factory import MCPToolsetFactory


SAMPLE_WEATHERAPI_CURRENT = {
    "location": {
        "name": "Wellington",
        "region": "Wellington",
        "country": "New Zealand",
        "lat": -41.3,
        "lon": 174.78,
        "localtime": "2026-11-23 00:15",
    },
    "current": {
        "temp_c": 15.4,
        "feelslike_c": 11.2,
        "humidity": 88,
        "condition": {"text": "Overcast", "icon": "//cdn.weatherapi.com/..."},
        "wind_kph": 26.5,
        "wind_dir": "S",
        "precip_mm": 0.0,
        "pressure_mb": 1025.0,
        "uv": 2.0,
        "air_quality": {
            "us-epa-index": 1,
            "pm2_5": 5.2,
            "pm10": 7.8,
        },
    },
}

SAMPLE_WEATHERAPI_FORECAST = {
    "location": {"name": "Wellington", "region": "Wellington", "country": "New Zealand"},
    "forecast": {
        "forecastday": [
            {
                "date": "2026-11-23",
                "day": {
                    "maxtemp_c": 18.0,
                    "mintemp_c": 12.5,
                    "avgtemp_c": 15.0,
                    "daily_chance_of_rain": 40,
                    "condition": {"text": "Scattered clouds"},
                    "maxwind_kph": 32.0,
                    "uv": 5.0,
                },
            },
            {
                "date": "2026-11-24",
                "day": {
                    "maxtemp_c": 16.5,
                    "mintemp_c": 11.0,
                    "avgtemp_c": 14.0,
                    "daily_chance_of_rain": 10,
                    "condition": {"text": "Sunny and breezy"},
                    "maxwind_kph": 28.0,
                    "uv": 6.0,
                },
            },
        ]
    },
}


def test_weather_config_properties_and_customization():
    """Verify WeatherConfig externalizes endpoints for implemented tools and supports custom overrides."""
    # Test default properties for implemented tools
    default_cfg = WeatherConfig(api_key="test-key")
    assert default_cfg.base_url == "https://api.weatherapi.com/v1"
    assert default_cfg.current_endpoint == "/current.json"
    assert default_cfg.forecast_endpoint == "/forecast.json"
    assert default_cfg.current_url == "https://api.weatherapi.com/v1/current.json"
    assert default_cfg.forecast_url == "https://api.weatherapi.com/v1/forecast.json"
    assert default_cfg.timeout_seconds == 10.0
    assert default_cfg.aqi_enabled is True

    # Test custom overrides for separate deployment / proxying
    custom_cfg = WeatherConfig(
        api_key="custom-key",
        base_url="https://internal-weather-gateway.company.com/v1",
        current_endpoint="/live-weather",
        forecast_endpoint="/live-forecast",
        timeout_seconds=5.0,
    )
    assert custom_cfg.current_url == "https://internal-weather-gateway.company.com/v1/live-weather"
    assert custom_cfg.forecast_url == "https://internal-weather-gateway.company.com/v1/live-forecast"
    assert custom_cfg.timeout_seconds == 5.0


@pytest.mark.asyncio
async def test_weather_client_missing_key_returns_error_no_fallback():
    """Verify client strictly requires WeatherAPI key and performs NO fallback when missing."""
    client = WeatherClient(api_key="")

    data = await client.get_current_weather("Wellington")
    assert "error" in data
    assert "WeatherAPI API key not configured" in data["error"]
    assert data["provider"] == "WeatherAPI.com"

    forecast = await client.get_weather_forecast("Wellington")
    assert "error" in forecast
    assert "WeatherAPI API key not configured" in forecast["error"]


@pytest.mark.asyncio
async def test_weather_client_current_weather_parsing():
    """Verify WeatherClient accurately normalizes WeatherAPI response."""
    client = WeatherClient(api_key="mock_test_key_123")

    class MockResponse:
        is_error = False
        status_code = 200
        def json(self): return SAMPLE_WEATHERAPI_CURRENT

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=MockResponse())):
        data = await client.get_current_weather("Wellington")

    assert data["provider"] == "WeatherAPI.com"
    assert data["city"] == "Wellington"
    assert data["country"] == "New Zealand"
    assert data["temperature_c"] == 15.4
    assert data["feelslike_c"] == 11.2
    assert data["condition"] == "Overcast"
    assert data["wind_kph"] == 26.5
    assert data["air_quality"]["us_epa_index"] == 1
    assert data["air_quality"]["air_quality_level"] == "Good (Minimal Impact)"


@pytest.mark.asyncio
async def test_weather_client_forecast_parsing():
    """Verify WeatherClient normalizes multi-day forecast."""
    client = WeatherClient(api_key="mock_test_key_123")

    class MockResponse:
        is_error = False
        status_code = 200
        def json(self): return SAMPLE_WEATHERAPI_FORECAST

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=MockResponse())):
        data = await client.get_weather_forecast("Wellington", days=2)

    assert data["provider"] == "WeatherAPI.com"
    assert data["city"] == "Wellington"
    assert len(data["forecast_days"]) == 2
    assert data["forecast_days"][0]["max_temp_c"] == 18.0
    assert data["forecast_days"][0]["chance_of_rain_percent"] == 40
    assert data["forecast_days"][1]["condition"] == "Sunny and breezy"


@pytest.mark.asyncio
async def test_weather_client_air_quality_parsing():
    """Verify WeatherClient extracts air quality indices correctly."""
    client = WeatherClient(api_key="mock_test_key_123")

    class MockResponse:
        is_error = False
        status_code = 200
        def json(self): return SAMPLE_WEATHERAPI_CURRENT

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=MockResponse())):
        data = await client.get_air_quality("Wellington")

    assert data["provider"] == "WeatherAPI.com"
    assert data["city"] == "Wellington"
    assert data["air_quality"]["us_epa_index"] == 1
    assert data["air_quality"]["pm2_5_ug_m3"] == 5.2
    assert data["uv_index"] == 2.0


@pytest.mark.asyncio
async def test_weather_client_api_error_handling():
    """Verify WeatherClient extracts error payload on WeatherAPI failure without falling back."""
    client = WeatherClient(api_key="mock_test_key_123")

    class MockErrorResponse:
        is_error = True
        status_code = 400
        reason_phrase = "Bad Request"
        def json(self):
            return {"error": {"code": 1006, "message": "No matching location found."}}

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=MockErrorResponse())):
        data = await client.get_current_weather("InvalidCityNameXYZ")

    assert "error" in data
    assert data["code"] == 1006
    assert data["error_category"] == "LOCATION_NOT_FOUND"
    assert "No matching location found" in data["error"]
    assert data["provider"] == "WeatherAPI.com"


@pytest.mark.asyncio
async def test_weather_client_swagger_error401_unauthorized():
    """Verify HTTP 401 (error401 in Swagger) invalid API key mapping."""
    client = WeatherClient(api_key="invalid_key")

    class Mock401Response:
        is_error = True
        status_code = 401
        reason_phrase = "Unauthorized"
        def json(self):
            return {"error": {"code": 2006, "message": "API key provided is invalid"}}

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=Mock401Response())):
        data = await client.get_current_weather("Wellington")

    assert data["status_code"] == 401
    assert data["code"] == 2006
    assert data["error_category"] == "UNAUTHORIZED"
    assert "API key provided is invalid" in data["message"]


@pytest.mark.asyncio
async def test_weather_client_swagger_error403_quota_exceeded():
    """Verify HTTP 403 (error403 in Swagger) quota exceeded mapping."""
    client = WeatherClient(api_key="quota_exhausted_key")

    class Mock403Response:
        is_error = True
        status_code = 403
        reason_phrase = "Forbidden"
        def json(self):
            return {"error": {"code": 2007, "message": "API key has exceeded calls per month quota."}}

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=Mock403Response())):
        data = await client.get_current_weather("Wellington")

    assert data["status_code"] == 403
    assert data["code"] == 2007
    assert data["error_category"] == "QUOTA_EXCEEDED"
    assert "quota" in data["hint"].lower()


@pytest.mark.asyncio
async def test_weather_client_swagger_error404_and_429_and_500():
    """Verify HTTP 404, 429, and 500 error code handling."""
    client = WeatherClient(api_key="valid_key")

    # 404
    class Mock404:
        is_error = True
        status_code = 404
        reason_phrase = "Not Found"
        url = httpx.URL("https://api.weatherapi.com/v1/invalid")
        def json(self): return {}

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=Mock404())):
        data = await client.get_current_weather("Wellington")
    assert data["status_code"] == 404
    assert data["error_category"] == "NOT_FOUND"

    # 429 Rate limited
    class Mock429:
        is_error = True
        status_code = 429
        reason_phrase = "Too Many Requests"
        def json(self): return {}

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=Mock429())):
        data = await client.get_current_weather("Wellington")
    assert data["status_code"] == 429
    assert data["error_category"] == "RATE_LIMITED"

    # 500 Server error
    class Mock500:
        is_error = True
        status_code = 500
        reason_phrase = "Internal Server Error"
        def json(self): return {"error": {"code": 9999, "message": "Internal application error."}}

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=Mock500())):
        data = await client.get_current_weather("Wellington")
    assert data["status_code"] == 500
    assert data["error_category"] == "SERVER_ERROR"
    assert data["code"] == 9999


@pytest.mark.asyncio
async def test_weather_client_timeout_and_connection_error():
    """Verify network timeout and connection error isolation."""
    client = WeatherClient(api_key="valid_key")

    with patch.object(httpx.AsyncClient, "get", side_effect=httpx.TimeoutException("Timeout")):
        data = await client.get_current_weather("Wellington")
    assert data["error_category"] == "TIMEOUT"
    assert data["status_code"] == 408

    with patch.object(httpx.AsyncClient, "get", side_effect=httpx.ConnectError("Connection refused")):
        data = await client.get_current_weather("Wellington")
    assert data["error_category"] == "CONNECTION_ERROR"
    assert data["status_code"] == 503


@pytest.mark.asyncio
async def test_weather_mcp_server_tool_schemas():
    """Verify WeatherService MCP server exports all 3 required tools."""
    tools = await weather_mcp_server.list_tools()
    tool_names = [t.name for t in tools]

    assert "get_current_weather" in tool_names
    assert "get_weather_forecast" in tool_names
    assert "get_air_quality" in tool_names

    current_tool = next(t for t in tools if t.name == "get_current_weather")
    assert "weather" in current_tool.description.lower()


@pytest.mark.asyncio
async def test_weather_mcp_server_tool_execution():
    """Verify executing get_current_weather via MCP Server returns valid JSON payload."""
    class MockResponse:
        is_error = False
        status_code = 200
        def json(self): return SAMPLE_WEATHERAPI_CURRENT

    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(return_value=MockResponse())):
        result = await weather_mcp_server.call_tool("get_current_weather", {"city": "Wellington"})

    assert not result.is_error
    assert len(result.content) > 0
    raw_json = result.content[0].text
    parsed = json.loads(raw_json)

    assert parsed["city"] == "Wellington"
    assert parsed["temperature_c"] == 15.4
    assert parsed["condition"] == "Overcast"
    assert parsed["provider"] == "WeatherAPI.com"


@pytest.mark.asyncio
async def test_mcp_factory_builds_weather_toolset():
    """Verify MCPToolsetFactory successfully builds ADK toolset for WeatherService."""
    class DummyServerRecord:
        id = "srv-weather-1"
        name = "WeatherService"
        description = "Live weather conditions"
        transport_type = "stdio"
        stdio_command = ".venv/bin/python mcp_servers/weather/server.py --stdio"
        auth_type = "none"
        auth_secret_encrypted = None
        action_scope = "read_only"
        allowed_tools = None
        blocked_tools = None
        cached_tool_schemas = [
            {"name": "get_current_weather", "description": "Current weather"},
            {"name": "get_weather_forecast", "description": "Forecast"},
        ]
        enabled = True

    factory = MCPToolsetFactory()
    toolsets = await factory.build_toolsets([DummyServerRecord()])

    assert len(toolsets) == 1
    # Verify toolset is an instance of Google ADK McpToolset
    from google.adk.tools.mcp_tool import McpToolset
    assert isinstance(toolsets[0], McpToolset)


@pytest.mark.asyncio
async def test_weather_mcp_registration_via_api_endpoint():
    """Verify registering WeatherService via REST API endpoint /mcp/servers."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from api.main import app
    from api.database import get_db
    from api.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    payload = {
        "name": "WeatherService",
        "description": "WeatherService registered via API endpoint test",
        "transport_type": "stdio",
        "stdio_command": ".venv/bin/python mcp_servers/weather/server.py --stdio",
        "auth_type": "none",
        "action_scope": "read_only",
        "tenant_id": "default",
    }

    mock_discovered_tools = [
        {"name": "get_current_weather", "description": "Current weather"},
        {"name": "get_weather_forecast", "description": "Forecast"},
        {"name": "get_air_quality", "description": "Air quality"},
    ]

    try:
        with patch("api.routes.admin._probe_mcp_server", new=AsyncMock(return_value=mock_discovered_tools)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                # 1. Test POST /mcp/servers (direct endpoint)
                resp = await client.post("/mcp/servers", json=payload)
                assert resp.status_code == 200
                data = resp.json()
                assert data["name"] == "WeatherService"
                assert data["tools_discovered"] == 3
                assert "get_current_weather" in data["tool_names"]

                # 2. Test idempotent re-registration / update via /mcp/servers
                resp2 = await client.post("/mcp/servers", json=payload)
                assert resp2.status_code == 200
                assert resp2.json()["name"] == "WeatherService"

                # 3. Test GET /mcp/servers
                list_resp = await client.get("/mcp/servers?tenant_id=default")
                assert list_resp.status_code == 200
                servers = list_resp.json()
                matching = [s for s in servers if s["name"] == "WeatherService"]
                assert len(matching) == 1
                assert matching[0]["action_scope"] == "read_only"
                assert matching[0]["tool_count"] == 3
    finally:
        app.dependency_overrides.clear()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

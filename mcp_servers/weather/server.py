"""
Weather Information Search MCP Server.
Exposes real-time atmospheric, forecast, and air quality tools via Anthropic MCP.
Strictly integrates with WeatherAPI.com using externalized configuration properties.
Supports Stdio (default for zero-network agent orchestration) and SSE transports.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure package paths are present on sys.path for standalone or subshell execution
_mcp_servers_root = Path(__file__).resolve().parent.parent
_project_root = _mcp_servers_root.parent
for _p in (_project_root, _mcp_servers_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mcp.server.mcpserver import MCPServer
from mcp_servers.weather.client import WeatherClient
from mcp_servers.weather.config import WeatherConfig, get_weather_config

# Initialize MCP Server instance
server = MCPServer(
    name="WeatherService",
    instructions=(
        "WeatherService provides real-time meteorological conditions, multi-day forecasts, "
        "wind speed, humidity, precipitation, and air quality intelligence for cities globally "
        "powered directly by WeatherAPI.com."
    ),
)

_client: Optional[WeatherClient] = None


def get_client() -> WeatherClient:
    global _client
    if _client is None:
        _client = WeatherClient()
    return _client


@server.tool(
    name="get_current_weather",
    description=(
        "Get current real-time atmospheric conditions for a given city from WeatherAPI.com. "
        "Returns temperature (Celsius), feels-like, humidity, wind speed (kph), "
        "wind direction, condition description, precipitation, pressure, UV index, and air quality."
    ),
)
async def get_current_weather(city: str) -> str:
    """
    Retrieve real-time weather metrics for a city.

    Args:
        city: The name of the city (e.g. 'Wellington', 'London', 'New York').
    """
    client = get_client()
    try:
        data = await client.get_current_weather(city)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to retrieve current weather for '{city}': {str(e)}",
            "city": city,
            "provider": "WeatherAPI.com",
        })


@server.tool(
    name="get_weather_forecast",
    description=(
        "Get multi-day weather forecast (up to 14 days) for a city from WeatherAPI.com. "
        "Returns daily max/min temperatures, precipitation probabilities, "
        "maximum wind speed, UV index, and expected conditions."
    ),
)
async def get_weather_forecast(city: str, days: int = 3) -> str:
    """
    Retrieve multi-day weather forecast.

    Args:
        city: The name of the city.
        days: Number of days to forecast (1 to 14, default: 3).
    """
    client = get_client()
    try:
        data = await client.get_weather_forecast(city, days=days)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to retrieve forecast for '{city}': {str(e)}",
            "city": city,
            "provider": "WeatherAPI.com",
        })


@server.tool(
    name="get_air_quality",
    description=(
        "Get air quality metrics and environmental indices for a city from WeatherAPI.com. "
        "Returns US EPA Index, PM2.5, PM10 pollution levels, and UV index."
    ),
)
async def get_air_quality(city: str) -> str:
    """
    Retrieve air quality metrics for a city.

    Args:
        city: The name of the city.
    """
    client = get_client()
    try:
        data = await client.get_air_quality(city)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to retrieve air quality for '{city}': {str(e)}",
            "city": city,
            "provider": "WeatherAPI.com",
        })


def main():
    parser = argparse.ArgumentParser(description="Weather Information MCP Server (WeatherAPI.com)")
    parser.add_argument("--stdio", action="store_true", default=True, help="Run over stdio (default)")
    parser.add_argument("--sse", action="store_true", help="Run as Server-Sent Events (SSE) server")
    parser.add_argument("--port", type=int, default=8001, help="Port for SSE server (default: 8001)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for SSE server (default: 127.0.0.1)")

    args = parser.parse_args()

    if args.sse:
        print(f"Starting WeatherService MCP server over SSE on http://{args.host}:{args.port}/sse ...", file=sys.stderr)
        server.run(transport="sse", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()

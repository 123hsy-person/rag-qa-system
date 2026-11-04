from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# 初始化fastmcp服务器
mcp = FastMCP("weather", log_level="ERROR")

# 常量
# 美国国家气象局的API基础地址
# 要请求的远程服务器域名。后面拼接不同路径，就能调用不同接口
NWS_API_BASE = "https://api.weather.gov"
# 请求头里的身份标识
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    # 请求天气数据，接受一个参数（请求天气数据时所用的URL）
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    # 内部使用http库调用指定的URL，返回从http中拿到的结果
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


# 对告警数据做格式化
def format_alert(feature: dict) -> str:
    props = feature["properties"]
    return f"""
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Severity: {props.get("severity", "Unknown")}
Description: {props.get("description", "No description available")}
Instructions: {props.get("instruction", "No specific instructions provided")}
"""


@mcp.tool()
async def get_alerts(state: str) -> str:
    # 获取美国某个州的天气预警，参数是美国的州代码
    """Get weather alerts for a US state.
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)  # 直接调用美国气象局的接口
    # 检查调用没有失败
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    # 检查调用的地区是否不存在预警信息，不存在立即返回结果
    if not data["features"]:
        return "No active alerts for this state."
    # 对预警信息做格式化
    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    # 参数是经纬度
    """Get weather forecast for a location.
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """ # 函数文档字符串，python里面三引号包裹写在函数定义第一行
    # 第一次调用接口，获取对应的天气预报办公室的信息
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)
    if not points_data:
        return "Unable to fetch forecast data for this location."

    # 提取对应天气预报的URL
    forecast_url = points_data["properties"]["forecast"]
    # 再次请求美国气象局，拿到天气预报
    forecast_data = await make_nws_request(forecast_url)
    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # 对天气预报格式化并且返回
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period["name"]}:
Temperature: {period["temperature"]}°{period["temperatureUnit"]}
Wind: {period["windSpeed"]} {period["windDirection"]}
Forecast: {period["detailedForecast"]}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

if __name__ == "__main__":
    mcp.run(transport='stdio')
# stdio表示mcpserver和cline的沟通方式是mcpserver的输入和输出


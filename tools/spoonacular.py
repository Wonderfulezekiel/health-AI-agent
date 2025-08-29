import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List

# Prefer langchain.tools.Tool, fall back to langchain_core.tools.Tool for compatibility
try:
    from langchain.tools import Tool  # type: ignore
except Exception:  # pragma: no cover
    from langchain_core.tools import Tool  # type: ignore


def _get_spoonacular_api_key() -> str:
    return os.getenv("SPOONACULAR_API_KEY", "")


def _fetch_spoonacular_meal_plan(query: str) -> Dict[str, Any]:
    api_key = _get_spoonacular_api_key()
    if not api_key:
        return {"results": [], "error": "missing_api_key"}

    base_url = "https://api.spoonacular.com/mealplanner/generate"
    params = {
        "timeFrame": "day",
        "targetCalories": query,
        "apiKey": api_key,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=20) as resp:  # nosec - public API request
        data = json.loads(resp.read().decode("utf-8"))
    return {"results": data.get("meals", [])}


def _spoonacular_tool_fn(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return json.dumps({"results": [], "error": "empty_query"})
    try:
        payload = _fetch_spoonacular_meal_plan(query)
        return json.dumps(payload)
    except Exception as e:  # pragma: no cover
        return json.dumps({"results": [], "error": str(e)})


def get_spoonacular_tools() -> List[Tool]:
    description = (
        "Use this tool to generate a meal plan for a day with a target number of calories. "
        "It returns a list of meals with their title, serving size, and ready time. "
        "Use it when a user asks for a meal plan or recipe suggestions."
    )
    return [
        Tool(
            name="spoonacular_meal_plan",
            description=description,
            func=_spoonacular_tool_fn,
        )
    ]

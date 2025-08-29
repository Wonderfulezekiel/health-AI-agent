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


def _build_user_agent() -> str:
    app_name = os.getenv("OFF_APP_NAME", "WonderfulHealthAssistant")
    app_version = os.getenv("OFF_APP_VERSION", "0.1")
    contact = os.getenv("OFF_CONTACT_EMAIL", "contact@example.com")
    return f"{app_name}/{app_version} ({contact})"


def _fetch_openfoodfacts_search(query: str) -> Dict[str, Any]:
    base_url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 1,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _build_user_agent()})
    with urllib.request.urlopen(req, timeout=20) as resp:  # nosec - public API request
        data = json.loads(resp.read().decode("utf-8"))
    products: List[Dict[str, Any]] = data.get("products", []) or []
    if not products:
        return {"results": []}

    p = products[0]
    result = {
        "product_name": p.get("product_name") or p.get("generic_name"),
        "brands": p.get("brands"),
        "nutri_score_grade": p.get("nutrition_grade_fr"),
        "nova_group": p.get("nova_group"),
        "nutriments": {
            "energy_kcal_100g": p.get("nutriments", {}).get("energy-kcal_100g"),
            "fat_100g": p.get("nutriments", {}).get("fat_100g"),
            "saturated_fat_100g": p.get("nutriments", {}).get("saturated-fat_100g"),
            "sugars_100g": p.get("nutriments", {}).get("sugars_100g"),
            "salt_100g": p.get("nutriments", {}).get("salt_100g"),
            "proteins_100g": p.get("nutriments", {}).get("proteins_100g"),
            "fiber_100g": p.get("nutriments", {}).get("fiber_100g"),
        },
        "additives_tags": p.get("additives_tags"),
        "ingredients_text": p.get("ingredients_text"),
        "url": p.get("url"),
        "image_url": p.get("image_url"),
        "code": p.get("code"),
        "source": "openfoodfacts",
    }
    return {"results": [result]}


def _openfoodfacts_tool_fn(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return json.dumps({"results": [], "error": "empty_query"})
    try:
        payload = _fetch_openfoodfacts_search(query)
        return json.dumps(payload)
    except Exception as e:  # pragma: no cover
        return json.dumps({"results": [], "error": str(e)})


def get_openfoodfacts_tools() -> List[Tool]:
    description = (
        "Use this tool to fetch nutrition facts from Open Food Facts for a food name or meal. "
        "It returns Nutri-Score, NOVA group, and per-100g nutrients when available. "
        "Use it when a user asks if a food is healthy or requests nutrition info."
    )
    return [
        Tool(
            name="open_food_facts_search",
            description=description,
            func=_openfoodfacts_tool_fn,
        )
    ] 
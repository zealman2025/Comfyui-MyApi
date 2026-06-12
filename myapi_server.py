"""MYAPI 设置 API（全局 API Key 读写）。"""

from aiohttp import web

from server import PromptServer

from .myapi_keys import api_key_store, PROVIDERS

API_PREFIX = "/myapi"


@PromptServer.instance.routes.get(f"{API_PREFIX}/settings/api_keys")
async def myapi_get_api_keys(request):
    return web.json_response(api_key_store.get_masked_all())


@PromptServer.instance.routes.post(f"{API_PREFIX}/settings/api_keys")
async def myapi_post_api_key(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"success": False, "error": "invalid json"}, status=400)

    provider = (data.get("provider") or "").strip()
    api_key = data.get("api_key", "")

    if provider not in PROVIDERS:
        return web.json_response({"success": False, "error": "invalid provider"}, status=400)

    if not api_key_store.set(provider, api_key):
        return web.json_response(
            {"success": False, "error": "empty api_key or save failed"},
            status=400,
        )

    return web.json_response({"success": True, "provider": provider})

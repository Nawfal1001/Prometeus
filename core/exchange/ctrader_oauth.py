# ============================================================
#  PROMETHEUS — cTrader OAuth helper
# ============================================================

from __future__ import annotations

from urllib.parse import urlencode
import aiohttp


TOKEN_URL = "https://openapi.ctrader.com/apps/token"
AUTHORIZE_URL = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"


def build_authorization_url(client_id: str, redirect_uri: str, scope: str = "trading", product: str = "web") -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "product": product,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_tokens(client_id: str, client_secret: str, code: str, redirect_uri: str) -> dict:
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(TOKEN_URL, data=payload) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                return {"status": "error", "http_status": resp.status, "response": data}
            return {"status": "ok", "response": data}


async def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> dict:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(TOKEN_URL, data=payload) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                return {"status": "error", "http_status": resp.status, "response": data}
            return {"status": "ok", "response": data}

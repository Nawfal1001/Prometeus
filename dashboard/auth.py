# ============================================================
#  PROMETHEUS — Dashboard Authentication
# ============================================================
#  Single-user session auth.
#  Disabled when DASHBOARD_USERNAME or DASHBOARD_PASSWORD is empty
#  (backward-compatible with existing deployments).
# ============================================================

import hmac
import time
from pathlib import Path

from fastapi import Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

import config.settings as cfg

BASE_DIR = Path(__file__).parent
_templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Paths that are always reachable without auth
_EXEMPT_PREFIXES = ("/login", "/logout", "/static/", "/favicon", "/health", "/ws")


def auth_enabled() -> bool:
    return bool(getattr(cfg, "DASHBOARD_USERNAME", "")) and bool(getattr(cfg, "DASHBOARD_PASSWORD", ""))


def _session_ttl_seconds() -> int:
    return int(float(getattr(cfg, "DASHBOARD_SESSION_TTL_HOURS", 24)) * 3600)


def _is_exempt(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in _EXEMPT_PREFIXES)


def _is_logged_in(request: Request) -> bool:
    session = request.scope.get("session")
    if not session:
        return False
    if not session.get("authed"):
        return False
    exp = session.get("exp", 0)
    if exp and time.time() > exp:
        session.clear()
        return False
    return True


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not auth_enabled():
            return await call_next(request)
        if _is_exempt(request.url.path):
            return await call_next(request)
        if _is_logged_in(request):
            return await call_next(request)
        # Browser navigation -> redirect to login. API call -> 401 JSON.
        accept = (request.headers.get("accept") or "").lower()
        wants_html = "text/html" in accept
        if wants_html:
            return RedirectResponse(url=f"/login?next={request.url.path}", status_code=303)
        return JSONResponse({"error": "authentication_required"}, status_code=401)


def register_auth_routes(app):
    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    async def login_form(request: Request):
        if not auth_enabled():
            return RedirectResponse(url="/", status_code=303)
        if _is_logged_in(request):
            return RedirectResponse(url="/", status_code=303)
        return _templates.TemplateResponse("login.html", {"request": request, "error": None, "next": request.query_params.get("next", "/")})

    @app.post("/login", response_class=HTMLResponse, include_in_schema=False)
    async def login_submit(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/")):
        if not auth_enabled():
            return RedirectResponse(url="/", status_code=303)
        expected_user = str(getattr(cfg, "DASHBOARD_USERNAME", ""))
        expected_pass = str(getattr(cfg, "DASHBOARD_PASSWORD", ""))
        ok_user = hmac.compare_digest(username.encode(), expected_user.encode())
        ok_pass = hmac.compare_digest(password.encode(), expected_pass.encode())
        if not (ok_user and ok_pass):
            logger.warning(f"[Auth] Failed login attempt for user={username!r}")
            return _templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password.", "next": next}, status_code=401)
        request.session["authed"] = True
        request.session["user"] = expected_user
        request.session["exp"] = time.time() + _session_ttl_seconds()
        logger.info(f"[Auth] User {expected_user!r} logged in")
        target = next if next.startswith("/") and not next.startswith("//") else "/"
        return RedirectResponse(url=target, status_code=303)

    @app.post("/logout", include_in_schema=False)
    @app.get("/logout", include_in_schema=False)
    async def logout(request: Request):
        try:
            request.session.clear()
        except Exception:
            pass
        return RedirectResponse(url="/login", status_code=303)

    @app.get("/api/auth/status", include_in_schema=False)
    async def auth_status(request: Request):
        return {"enabled": auth_enabled(), "logged_in": _is_logged_in(request), "user": request.session.get("user") if _is_logged_in(request) else None}

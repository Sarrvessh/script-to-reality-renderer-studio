"""Application implementation - ASGI."""

import math
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.controllers import base
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """Handle API-process startup recovery and shutdown logging in one place."""
    logger.info("startup event")

    configured_api_key = config.app.get("api_key", "")
    if configured_api_key in (None, ""):
        logger.warning(
            "API key authentication is disabled; keep the API on a trusted network"
        )
    elif isinstance(configured_api_key, str):
        # Log only the protection scope. Do not emit the key, its length, or a digest, so credentials do not enter the logging system.
        logger.info("API key authentication is enabled for /api/v1 and /tasks")
    else:
        logger.error(
            "API key authentication is misconfigured: app.api_key must be a string"
        )

    # Cross-posting runs on this process's thread pool and is not resumed after a service restart. At startup, converge Redis
    # activity whose executing process is confirmed gone into failed, so the task is not undeletable forever.
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()

    # Redis queue entries persist across API restarts. No task_done callback
    # exists in the new process to dispatch them, so fill its worker slots now.
    from app.controllers.manager.redis_manager import RedisTaskManager
    from app.controllers.v1 import video as video_controller

    if isinstance(video_controller.task_manager, RedisTaskManager):
        video_controller.task_manager.resume_queued_tasks()
    try:
        yield
    finally:
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    # Rejected inputs can contain NaN/Infinity, and custom validators attach
    # exception objects to ctx. Neither can be emitted by JSONResponse directly.
    errors = jsonable_encoder(
        e.errors(),
        custom_encoder={
            float: lambda value: value if math.isfinite(value) else str(value),
            Exception: str,
        },
    )
    return JSONResponse(
        status_code=400,
        content=utils.get_response(
            status=400, data=errors, message="field required"
        ),
    )


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize_allowed_origin(raw_origin: str) -> str | None:
    """Fold a config entry into the Origin form a browser actually sends.

    RFC 6454 fixes the Origin header as ``scheme://host[:port]``: no path, no trailing
    slash, scheme and host already lowercased, and the protocol's default port omitted. A user who copies
    ``https://frontend.example/`` from the address bar, or who writes the default port explicitly as
    ``https://frontend.example:443``, does not match the browser's actual
    ``https://frontend.example`` character for character, so the allowlist fails silently: the frontend only sees
    a CORS error, the server only leaves one blocked log line, and neither side points at the config itself.

    ``None`` means this spelling cannot match any Origin (missing scheme, scheme is not
    http/https, empty or whitespace hostname, malformed IPv6 literal, or an illegal or out-of-range port). The caller should
    drop it and warn. A parse failure must collapse to ``None``: this function runs at import time, and letting
    ``urlsplit()``'s ``ValueError`` escape means the whole API cannot start.
    """

    if raw_origin == "*":
        return raw_origin

    try:
        parsed = urlsplit(raw_origin)
        scheme = parsed.scheme.lower()
        # netloc is validated only when hostname / port are read: a malformed IPv6 literal is raised by urlsplit(),
        # and a non-numeric port or a port outside 0-65535 is raised when port is read.
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None

    # Only http/https have browser Origin semantics, and both have a default port that may be omitted.
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if any(character.isspace() for character in host):
        return None

    # Lowercase explicitly, and do not rely on urlsplit's case handling differing across Python versions.
    host = host.lower()
    # Browsers keep brackets around an IPv6 literal when they serialize Origin; default-port spellings such as ``:443``, ``:0443``,
    # and ``:80`` must be folded into the form without a port before they can match the real frontend.
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def parse_cors_allowed_origins(raw_origins: str | None) -> list[str]:
    """Parse the browser cross-origin allowlist.

    CORS only constrains cross-origin JavaScript in a browser. It does not affect curl, Postman, n8n,
    or a server-side SDK. When unset, return an empty list, which means cross-origin access stays closed; if the user really
    deployed a separate web frontend, open it explicitly with ``CORS_ALLOWED_ORIGINS``.

    Each origin is first folded into the canonical Origin header form, so ``https://a.example/``,
    ``HTTPS://A.Example``, and an explicit default port ``https://a.example:443`` produce the same
    result. A spelling that cannot match any Origin is dropped with a warning. A malformed entry drops only itself
    and does not stop parsing here. This function runs at import time, and a parse exception would keep the API from starting.
    """

    if not raw_origins:
        return []

    # Strip whitespace around comma-separated items and ignore empty items, so a common environment-variable form
    # ``https://a.example, https://b.example,`` does not produce an origin that can never match.
    origins: list[str] = []
    for candidate in raw_origins.split(","):
        item = candidate.strip()
        if not item:
            continue
        origin = _normalize_allowed_origin(item)
        if origin is None:
            logger.warning(
                f"ignoring configured CORS origin that cannot match a browser "
                f"Origin header: {item!r}"
            )
            continue
        if origin not in origins:
            origins.append(origin)
    return origins


def configure_cors(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configure CORS from an explicit allowlist; an empty list keeps the default same-origin policy."""

    if not allowed_origins:
        logger.info(
            "browser cross-origin API access is disabled; set "
            "CORS_ALLOWED_ORIGINS to enable trusted origins"
        )
        return

    allow_all_origins = "*" in allowed_origins
    configured_api_key = config.app.get("api_key", "")
    if allow_all_origins and configured_api_key in (None, ""):
        # ``*`` is a compatibility mode the user chose explicitly, so startup is not refused; but without authentication
        # it lets any web page read and call the API, so a locatable security warning must be left.
        logger.warning(
            "CORS allows every browser origin while API key authentication is "
            "disabled; configure app.api_key or restrict CORS_ALLOWED_ORIGINS"
        )

    instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # When ``*`` and credentials are both enabled, Starlette reflects any Origin.
        # Wildcard mode does not need cookie auth, so credentials are turned off on purpose; explicit origins
        # keep the old behavior so existing standalone web frontends are not broken in their credentials request mode.
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # When a remote HTTPS frontend reaches an API on localhost or a LAN, modern browsers also send
        # a Private Network Access preflight. Only an exact allowlisted origin may be permitted;
        # wildcard mode keeps refusing, so an arbitrary site cannot probe the user's private-network services.
        allow_private_network=not allow_all_origins,
    )


def is_browser_origin_allowed(
    request: Request, allowed_origins: list[str]
) -> bool:
    """Return whether a browser request origin is same-origin or on the explicit allowlist."""

    origin = request.headers.get("origin")
    if not origin:
        # curl, Postman, n8n, and server-side SDKs usually do not send Origin. Keep those requests
        # so a security fix does not change the call contract of existing API clients.
        return True
    if "*" in allowed_origins or origin in allowed_origins:
        return True

    # A browser may also send Origin on a same-origin POST. Compare only scheme + authority, and ignore
    # path and query; if a reverse-proxy deploy does not forward the public scheme/host correctly, declare the external origin
    # explicitly with CORS_ALLOWED_ORIGINS instead of trusting forwarded headers.
    request_url = urlsplit(str(request.url))
    request_origin = f"{request_url.scheme}://{request_url.netloc}"
    return origin == request_origin


def configure_browser_access(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configure server-side Origin protection and the browser CORS response policy together."""

    @instance.middleware("http")
    async def reject_untrusted_browser_origin(request: Request, call_next):
        """Actively reject an untrusted browser origin, including simple requests that do not need a CORS preflight."""

        if not is_browser_origin_allowed(request, allowed_origins):
            origin = request.headers.get("origin", "")
            logger.warning(
                f"blocked untrusted browser origin: method={request.method}, "
                f"path={request.url.path}, origin={origin}"
            )
            return JSONResponse(
                status_code=403,
                content=utils.get_response(
                    status=403,
                    message="cross-origin browser request is not allowed",
                ),
            )

        return await call_next(request)

    # Registered last, CORS middleware sits outside Origin protection: a trusted preflight can succeed directly,
    # an untrusted preflight is rejected by CORS, and a real request that needs no preflight still hits the 403 guard above.
    configure_cors(instance, allowed_origins)


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()


@app.middleware("http")
async def protect_generated_task_files(request: Request, call_next):
    """Protect the task-output static route so downloads cannot bypass API auth.

    ``/tasks`` is mounted separately with StaticFiles and cannot reuse APIRouter dependencies,
    so the middleware calls the same verify_token. The auth function allows the request when
    api_key is unset; OPTIONS preflight requests are also left for the CORS middleware.
    """

    request_path = request.url.path
    is_task_file = request_path == "/tasks" or request_path.startswith("/tasks/")
    if is_task_file and request.method != "OPTIONS":
        try:
            base.verify_token(request)
        except HttpException as exception:
            return exception_handler(request, exception)

    return await call_next(request)


# Follow the browser same-origin policy by default; open cross-origin only when the user explicitly configures trusted web origins.
# The local Script-to-Reality Renderer Studio UI is served by Vite on 5173 and proxied to this API.
# Trusting that loopback origin lets the studio save settings. Other websites stay blocked.
cors_allowed_origins = parse_cors_allowed_origins(
    os.getenv("CORS_ALLOWED_ORIGINS", "")
)
for _local_ui_origin in ("http://127.0.0.1:5173", "http://localhost:5173"):
    if _local_ui_origin not in cors_allowed_origins:
        cors_allowed_origins.append(_local_ui_origin)
configure_browser_access(app, cors_allowed_origins)

task_dir = utils.task_dir()
app.mount("/tasks", StaticFiles(directory=task_dir, html=True), name="")

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")

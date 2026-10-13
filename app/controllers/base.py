import secrets
from typing import Annotated
from uuid import uuid4

from fastapi import Header, Request

from app.config import config
from app.models.exception import HttpException

MAX_TASK_ID_LENGTH = 128


def normalize_task_id(value: object) -> str:
    """Return a log-safe request ID, replacing invalid client input with a UUID."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TASK_ID_LENGTH
        or not value.isprintable()
    ):
        return str(uuid4())
    return value


def get_task_id(request: Request) -> str:
    return normalize_task_id(request.headers.get("x-task-id"))


def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    return api_key


def get_api_key_values(request: Request) -> list[str]:
    """Return every API key header on the request, keeping duplicates for the security check."""

    # Starlette Headers provides getlist(), which can tell apart duplicate headers sent by a proxy or a client.
    # Lightweight Request stand-ins in unit tests use a plain dict, so the compatible fallback stays.
    get_list = getattr(request.headers, "getlist", None)
    if callable(get_list):
        return [value for value in get_list("x-api-key") if isinstance(value, str)]

    api_key = get_api_key(request)
    return [api_key] if isinstance(api_key, str) else []


def verify_token(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
):
    """Decide from configuration whether to validate the API key.

    An empty key keeps the existing local unauthenticated mode. After an administrator explicitly configures a non-empty key, API
    routes and task artifact downloads both require the client to send that same value in the ``x-api-key`` header. Declaring the
    parameter also shows the header in Swagger, which makes debugging easier in protected environments.
    """

    configured_key = config.app.get("api_key", "")
    if configured_key in (None, ""):
        return None

    # The setting must be a string. Lists, numbers, and other wrong types are rejected here so implicit string
    # conversion cannot create hard-to-spot authentication behavior; the error message also omits the actual key.
    if not isinstance(configured_key, str):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=500,
            message="API authentication is misconfigured",
        )

    # The FastAPI parameter declares x-api-key in OpenAPI; the real check always reads the Request,
    # so it can detect the same header sent more than once. Ordinary clients and reverse proxies may order duplicate headers
    # differently, so the request must be rejected instead of silently using the first or last value.
    token_values = get_api_key_values(request)
    if not token_values and isinstance(x_api_key, str):
        token_values = [x_api_key]

    if len(token_values) != 1:
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    # compare_digest supports only ASCII for str. Request headers are untrusted input, and an attacker
    # can send Latin-1 characters to trigger TypeError. Encoding both sides as UTF-8 bytes keeps the
    # constant-time comparison and also accepts a legal Unicode key in TOML.
    token = token_values[0]
    if not secrets.compare_digest(
        token.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    return None

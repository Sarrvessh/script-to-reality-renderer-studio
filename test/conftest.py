import pytest


@pytest.fixture(autouse=True)
def _no_llm_retry_sleep(monkeypatch):
    """Retry backoff is real in production; tests skip the wait."""
    from app.services import llm

    monkeypatch.setattr(llm, "_sleep_before_retry", lambda attempt, message="": None)

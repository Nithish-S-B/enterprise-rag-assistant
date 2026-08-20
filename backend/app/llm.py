"""Minimal OpenRouter client for text generation."""
import json
import os

import requests
from dotenv import load_dotenv


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
MAX_DEBUG_BODY_LENGTH = 1_000


def _get_configuration() -> tuple[str, str, str]:
    """Load and validate the non-secret OpenRouter connection settings."""
    load_dotenv(ENV_PATH)

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).rstrip("/")

    missing = []
    if not api_key:
        missing.append("OPENROUTER_API_KEY")
    if not model:
        missing.append("OPENROUTER_MODEL")
    if missing:
        raise RuntimeError(
            "OpenRouter configuration is missing: " + ", ".join(missing) + "."
        )

    return api_key, model, base_url


def _safe_response_body(response: requests.Response, api_key: str) -> object:
    """Return a truncated response body suitable for error diagnostics."""
    try:
        body = response.json()
    except ValueError:
        body = response.text

    body_text = json.dumps(body, ensure_ascii=False, default=str)
    body_text = body_text.replace(api_key, "[REDACTED]")
    if len(body_text) > MAX_DEBUG_BODY_LENGTH:
        body_text = body_text[:MAX_DEBUG_BODY_LENGTH] + "..."
    return body_text


def _error_message(body: object) -> str | None:
    """Extract a useful message from common OpenRouter error payload shapes."""
    if not isinstance(body, dict):
        return None

    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    if isinstance(body.get("message"), str):
        return body["message"]
    return None


def generate_text(prompt: str) -> str:
    """Send a prompt to the configured OpenRouter model and return its text."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty.")

    api_key, model, base_url = _get_configuration()
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"OpenRouter request failed before receiving a response: {type(error).__name__}."
        ) from error

    if not 200 <= response.status_code < 300:
        try:
            error_body = response.json()
        except ValueError:
            error_body = None

        message = _error_message(error_body)
        safe_body = _safe_response_body(response, api_key)
        detail = message or f"response body: {safe_body}"
        raise RuntimeError(
            f"OpenRouter request failed with HTTP status {response.status_code}: {detail}"
        )

    try:
        response_body = response.json()
    except ValueError as error:
        raise RuntimeError(
            "OpenRouter returned a non-JSON success response: "
            f"{_safe_response_body(response, api_key)}"
        ) from error

    if not isinstance(response_body, dict):
        raise RuntimeError(
            "OpenRouter returned an unexpected success response structure: "
            f"{_safe_response_body(response, api_key)}"
        )

    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "OpenRouter success response is missing a non-empty choices list: "
            f"{_safe_response_body(response, api_key)}"
        )

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(
            "OpenRouter success response has an invalid first choice: "
            f"{_safe_response_body(response, api_key)}"
        )

    message = first_choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError(
            "OpenRouter success response is missing choices[0].message.content: "
            f"{_safe_response_body(response, api_key)}"
        )

    content = message["content"]

    if not content.strip():
        raise RuntimeError("OpenRouter returned an empty generated response.")
    return content

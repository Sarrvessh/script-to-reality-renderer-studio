import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.utils import utils

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]*:[^/\s?#@]*@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)
_SECRET_TOKEN_RE = re.compile(r"\b(?:sk-or-v1-|sk-or-|sk-)[A-Za-z0-9_\-]{6,}\b")
_OPENROUTER_TIMEOUT_SECONDS = 120


DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Expert Video Scriptwriter and Factual Storyteller

You are a professional video scriptwriter, factual researcher, and audience-retention specialist creating high-quality scripts for AI-generated videos, Instagram Reels, YouTube Shorts, and long-form content.

Your objective is to produce factually accurate, original, engaging, and naturally spoken narration that delivers genuine value without clickbait, filler, or recognizable AI-writing clichés.

## 1. Factual Accuracy

Factual correctness takes priority over entertainment.

- Use established, verifiable information.
- Never fabricate facts, statistics, studies, quotes, dates, discoveries, or events.
- Distinguish confirmed facts from speculation, estimates, and disputed claims.
- Do not present uncertain information as established truth.
- For science, technology, history, medicine, finance, and current events, maintain technical and contextual accuracy.
- Simplify complex topics without introducing misleading explanations.
- If reliable verification is unavailable, avoid unsupported specifics or qualify uncertainty naturally.
- Never claim to have researched or verified information unless research tools or verified sources were actually available.
- Do not invent facts to make a hook more exciting.

## 2. Strong, Original Opening Hook

Begin immediately with a compelling, subject-specific hook.

The first sentence should establish curiosity, relevance, tension, an unexpected fact, or an intriguing question.

Choose the most natural hook style:
- A surprising but verified fact.
- An unexpected contradiction.
- A meaningful question.
- A compelling real-world consequence.
- An unusual observation.
- An interesting event already in progress.
- A genuine misconception worth correcting.

The hook must be directly relevant to the video's central subject.

Avoid generic openings such as:
"Did you know..."
"What if I told you..."
"Imagine a world where..."
"You won't believe..."
"Here's something crazy..."
"In today's world..."
"Let's dive into..."
"This changes everything..."

Never use misleading claims, exaggerated suspense, or fabricated surprises.

## 3. Narrative Structure

Build a coherent narrative with four natural stages:

HOOK:
Immediately establish why the subject is interesting or important.

DEVELOPMENT:
Explain the subject progressively using accurate facts, relevant examples, and clear transitions.

PAYOFF:
Deliver the main discovery, explanation, answer, or insight promised by the opening.

ENDING:
Conclude with a meaningful takeaway, consequence, or satisfying resolution.

Do not explicitly label these stages in the output.

For short-form videos, focus on one central idea rather than covering too many unrelated concepts.

For longer videos, allow deeper explanations while maintaining narrative momentum.

## 4. Audience Retention

Use retention techniques naturally and selectively:

- Introduce useful new information throughout the narration.
- Build curiosity through genuine unanswered questions.
- Resolve questions rather than creating endless suspense.
- Use concrete examples to explain abstract concepts.
- Introduce meaningful contrasts and discoveries.
- Vary sentence length and rhythm.
- Make transitions feel logically connected.
- Deliver small informational payoffs throughout the video.
- Keep the narrative moving without rushing complex explanations.

Every sentence must contribute information, context, clarity, or meaningful narrative progression.

Do not manufacture drama, artificially delay answers, or use repetitive cliffhangers.

## 5. Human-Like Narration

Write specifically for spoken delivery.

- Use clear, conversational language.
- Prefer natural sentence structures.
- Vary sentence length for realistic speech.
- Avoid overly formal, robotic, or theatrical wording.
- Explain technical terminology when necessary.
- Use analogies only when they genuinely improve understanding.
- Avoid excessive adjectives, forced humor, and unnecessary metaphors.
- Eliminate filler, repetition, and generic motivational statements.
- Make the narration sound like a knowledgeable human explaining something genuinely interesting.

Avoid formulaic phrases such as:
"But here's the twist..."
"And that's not all..."
"The shocking truth is..."
"Little did they know..."
"Fast forward to today..."
"This is a game changer..."
"The possibilities are endless..."

Do not use artificial enthusiasm or sensationalism.

## 6. Duration and Pacing

Adapt narration density to the requested video duration.

Use approximately 130–160 spoken words per minute for English, adjusting for complexity, delivery, and language.

For short videos:
- Establish the subject immediately.
- Prioritize concise, high-value information.
- Avoid unnecessary background explanations.
- Maintain a single coherent narrative.
- Deliver the payoff before the ending.

For longer videos:
- Develop context progressively.
- Explain important concepts thoroughly.
- Use examples where useful.
- Maintain narrative continuity.

Never sacrifice factual accuracy or clarity merely to reach a word count.

## 7. Visual Storytelling Compatibility

Write narration that can naturally support AI-generated visuals, real footage, diagrams, animation, or demonstrations.

- Prefer concrete descriptions where appropriate.
- Describe events and processes in a visually understandable way.
- Maintain logical continuity between ideas.
- Avoid vague or abstract filler that is difficult to illustrate.
- Do not invent visual evidence or describe fictional recreations as authentic footage.

Do not include scene descriptions, camera directions, visual prompts, editing instructions, timestamps, or sound effects.

Output narration only.

## 8. Language Adaptation

Respond in the same language as the video subject unless explicitly instructed otherwise.

Use natural grammar, vocabulary, and conversational rhythm appropriate to the language.

Avoid awkward literal translations and unnecessary code-switching.

Match the subject's seriousness, complexity, and intended audience.

## 9. Internal Quality Validation

Before responding, silently review the script for:

1. Factual correctness and uncertainty.
2. Hook originality and relevance.
3. Narrative coherence.
4. Natural spoken delivery.
5. Information density.
6. Audience retention without manipulation.
7. Absence of AI-generated clichés.
8. Appropriate duration and pacing.
9. A satisfying, meaningful payoff.
10. Compliance with all output constraints.

Revise internally if necessary.

## 10. Strict Output Contract

These requirements are mandatory.

1. Return only the final narration as plain text.
2. Follow the exact number of paragraphs specified by the user or application.
3. Do not include a title, heading, Markdown, numbering, bullets, or formatting.
4. Separate paragraphs with exactly one blank line.
5. Begin directly with the opening hook.
6. Do not include labels such as "Narrator", "Voiceover", "Hook", "Scene", or "Conclusion".
7. Do not mention this prompt, these instructions, or the script-generation process.
8. Do not mention the number of paragraphs or lines.
9. Do not include citations, research notes, explanations, or production instructions.
10. Do not include generic greetings or unnecessary introductions.
11. Do not include calls to like, comment, share, follow, or subscribe unless explicitly requested.
12. Do not add content before or after the narration.
13. If the requested duration and paragraph count conflict, preserve the required paragraph count while keeping the narration as close as practical to the duration.
14. Never invent information to satisfy length requirements.

## Core Principle

Earn the viewer's attention through accuracy, originality, clarity, and meaningful storytelling.

Create narration that is interesting because the subject is genuinely compelling, not because the viewer is manipulated by artificial suspense.

Return only the finished script.
""".strip()


# Claude Code CLI defaults to the coding agent's system prompt, whose many constraints are unrelated
# to copywriting and would pull script and keyword generation off brief, so the call replaces it entirely.
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are a factual video scriptwriter. Follow the user's instructions and output "
    "format exactly. Do not add a title, labels, or commentary. Output nothing else."
)
CLAUDE_CODE_DEFAULT_TIMEOUT = 300.0
# `--tools ""` disables all built-in tools, and `--safe-mode` disables CLAUDE.md, skills, hooks,
# plugins, MCP, and every other user-level customization, while auth, model choice, and permissions keep working.
# Both need a newer CLI; older builds exit with "unknown option", which the caller turns into a clear message.
CLAUDE_CODE_MIN_CLI_VERSION = "2.1.260"
# These environment variables make the CLI switch to an API key or a third-party provider (Bedrock, Vertex, Foundry,
# Mantle, Gateway, and so on), bypassing subscription login and adding extra charges. Listing them one by one is easy to miss,
# and the CLI will add more providers later, so whole prefix classes are stripped:
#   ANTHROPIC_*           API key, auth token, base URL, each provider endpoint, and profile
#   CLAUDE_CODE_USE_*     provider switches
#   CLAUDE_CODE_SKIP_*_AUTH  switches that skip provider authentication
CLAUDE_CODE_CONFLICTING_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
CLAUDE_CODE_CONFLICTING_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_GATEWAY_TOKEN_FILE_DESCRIPTOR",
)
# These two kinds of variables must not be stripped:
#   CLAUDE_CODE_OAUTH_TOKEN is the only subscription auth method inside a container (it does not match the prefixes above);
#   *_CONFIG_DIR only points at where credentials are stored; stripping it would invalidate an already logged-in subscription.
CLAUDE_CODE_PRESERVED_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CONFIG_DIR",
)


def _is_conflicting_claude_code_env(name: str) -> bool:
    """Return whether an environment variable would switch the CLI from subscription login to another auth method."""
    if name in CLAUDE_CODE_PRESERVED_ENV_VARS:
        return False
    if name in CLAUDE_CODE_CONFLICTING_ENV_VARS:
        return True
    if name.startswith(CLAUDE_CODE_CONFLICTING_ENV_PREFIXES):
        return True
    return name.startswith("CLAUDE_CODE_SKIP_") and name.endswith("_AUTH")


def coerce_claude_code_timeout(value, config_key: str = "claude_code_timeout"):
    """
    Parse the configured timeout into a positive finite number of seconds.

    TOML may store `claude_code_timeout = 300` (int/float) or
    `"300"` (a string), so `strip()` cannot be called directly. nan / inf would make
    `subprocess.run(timeout=...)` block forever, so those values are rejected here as well.
    """
    if value is None:
        return CLAUDE_CODE_DEFAULT_TIMEOUT

    if isinstance(value, bool):
        # bool is a subclass of int, but True seconds is obviously not the timeout the user intended.
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return CLAUDE_CODE_DEFAULT_TIMEOUT
        try:
            seconds = float(text)
        except ValueError:
            raise ValueError(
                f"{config_key} must be a number of seconds, got {value!r}"
            ) from None
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if not math.isfinite(seconds):
        raise ValueError(f"{config_key} must be a finite number, got {value!r}")
    if seconds <= 0:
        raise ValueError(f"{config_key} must be greater than 0, got {value!r}")
    return seconds


def _resolve_provider_field_value(raw_value, default_value):
    """
    Fall back to the Registry default only when the value is "not configured".

    The previous `raw or default_value` also treated legal values such as 0 and false as unset
    and replaced them: `claude_code_timeout = 0` was silently changed to 300, while `"0"` raised. The default
    applies only for None or a blank string, so validation stays consistent for every spelling.
    """
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and not raw_value.strip():
        return default_value
    return raw_value


def build_claude_code_env(base_env=None):
    """
    Build a subprocess environment that depends only on subscription login.

    Returns (environment dict, list of removed variable names). Removed variables would switch
    auth method or provider. `CLAUDE_CODE_OAUTH_TOKEN` must be kept: a container has no keychain, so the CLI can
    complete subscription auth only with that token.
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = sorted(name for name in env if _is_conflicting_claude_code_env(name))
    for name in removed:
        env.pop(name, None)
    return env, removed


def _normalize_text_response(content, llm_provider: str) -> str:
    # Different LLM SDKs, on failure or when blocked, may return None, an empty string,
    # or even a non-string object. Validate that here so a later direct call to
    # `.replace()` does not raise an attribute error such as `NoneType`.
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # Reasoning models such as MiniMax M3 and DeepSeek R1 may wrap internal reasoning in
    # `<think>...</think>`. Video scripts and keywords need only the final speakable text.
    # If the service layer does not strip it, the WebUI, subtitles, and voiceover all treat the thinking as body text.
    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # The preceding ``strip()`` already cleared leading and trailing whitespace. Single newlines and
    # double newlines in the body must be kept: script generation uses double newlines to separate paragraphs. Subtitle processing also reads the user's copy line by line.
    return content


def _sanitize_error_message(error: object) -> str:
    """
    Clean error text returned to the WebUI/API so credentials in a custom base_url are not leaked.

    Some OpenAI-compatible SDKs paste the request URL into the exception. If a user configured
    `https://user:pass@example.com/v1` for a proxy gateway, returning `str(e)` directly
    would expose the password to the page, API callers, or later logs. Only the error text is rewritten, not the
    actual request URL, so the normal call path is unchanged.
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    message = _SECRET_TOKEN_RE.sub("[redacted]", message)
    return message


def _classify_openrouter_error(error: object, message: str) -> str:
    """Turn OpenRouter/OpenAI SDK failures into a short English cause.

    The original sanitized message is kept so logs stay diagnosable, but the
    prefix names the usual causes: a bad key, a free-tier rate limit, a
    timeout, or a model ID that OpenRouter no longer serves.
    """
    status = getattr(error, "status_code", None)
    name = type(error).__name__
    lowered = message.lower()
    if status in (401, 403) or name in {"AuthenticationError", "PermissionDeniedError"}:
        summary = "invalid API key or unauthorized request"
    elif status == 429 or name == "RateLimitError":
        summary = (
            "rate limit exceeded; wait and retry, or set openrouter_model_name "
            "to another current free model"
        )
    elif name in {"APITimeoutError", "APIConnectionError"} or "timed out" in lowered:
        summary = "request timed out or the provider connection failed"
    elif (
        status == 404
        or name == "NotFoundError"
        or ("model" in lowered and "not found" in lowered)
    ):
        summary = (
            "model is unavailable; leave the model empty to use openrouter/free "
            "or choose another ID from https://openrouter.ai/models?variant=free"
        )
    elif status in (400, 422):
        summary = "the model rejected the request"
    else:
        return message
    return f"{summary}. {message}"


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # An OpenAI-compatible API may, on failure, return a response with no choices,
    # or with empty choices/message/content.
    # Validate the structure here to avoid low-level attribute errors such as
    # `NoneType is not subscriptable`.
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """Read a field from either a dict or an SDK response object."""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _extract_qwen_generation_text(response) -> str:
    """
    Extract text from a DashScope Generation response.

    When Qwen is called with `messages` it returns a chat structure:
    `output.choices[0].message.content`; only the older completion shape returns
    `output.text`. Both paths are accepted so a None `output.text` does not
    continue into `.replace()` and raise an undiagnosable AttributeError.
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(prompt: str, app_config=None) -> str:
    sdk_client = None
    sdk_stream = None
    llm_provider = ""
    try:
        # The WebUI lets the user prepare the next piece of copy while a video is generating. The caller may pass a snapshot
        # of the config at submit time so retries do not, after a background task finishes and applies new config,
        # switch to another Provider, base URL, or model.
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = str(
            runtime_app_config.get(provider.config_key("api_key"), "") or ""
        ).strip()
        # config.toml remains the primary secret store. An empty OpenRouter key
        # falls back to the environment so the key does not have to be written
        # into a file. The value is never logged.
        if not api_key and llm_provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama's default address depends on whether this process is in a container, so it cannot be stored as a static Registry
        # value; the Registry still owns the model and required-field rules, and the runtime difference is resolved here.
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: _resolve_provider_field_value(
                runtime_app_config.get(provider.config_key(field.config_suffix)),
                field.default_value,
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            response = dashscope.Generation.call(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # The newer google-genai SDK exposes model services through a unified Client. The context manager
                # closes the underlying HTTP connection after the request so frequent generation does not pile up connections.
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Cloudflare's currently recommended AI Gateway REST API is compatible with the OpenAI SDK.
            # The account ID builds the unified endpoint, and the gateway ID is selected by a request header; this path
            # no longer calls the Workers AI /ai/run/{model} dedicated API.
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # The Azure OpenAI SDK builds a dedicated request URL from `azure_endpoint` and `api_version`,
            # so it cannot reuse the ordinary OpenAI-compatible `base_url` setup below.
            # Finish the request inside the Azure branch and return immediately so a later fallback does not
            # overwrite the client and leave the user's validated Azure credentials unused on the real request.
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = sdk_client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter == "claude_code":
            # A Claude subscription (Pro / Max / Team) does not issue an API key; its credentials can be used
            # only by the official Claude Code client. This does not call the Anthropic API directly.
            # It calls the locally logged-in claude CLI in headless mode (`claude -p`),
            # lets the CLI handle auth, and script generation consumes only the text it returns.
            configured_cli = (extra_values.get("cli_path") or "").strip() or "claude"
            cli_path = shutil.which(configured_cli)
            if not cli_path and os.path.isfile(configured_cli):
                cli_path = configured_cli
            if not cli_path:
                raise ValueError(
                    f"{llm_provider}: claude CLI not found ('{configured_cli}'), "
                    f"install it in the runtime or set "
                    f"{provider.config_key('cli_path')} in the config.toml file."
                )

            try:
                timeout_seconds = coerce_claude_code_timeout(
                    extra_values.get("timeout"), provider.config_key("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            # The prompt is passed on stdin, not on the command line: the npm-installed claude on Windows
            # is claude.cmd, and cmd.exe truncates arguments at the first newline, so a multiline
            # prompt and the isolation arguments after it would both be lost.
            command = [
                cli_path,
                "-p",
                "--output-format",
                "json",
                "--system-prompt",
                CLAUDE_CODE_SYSTEM_PROMPT,
                # Disable all built-in tools so this only generates text.
                "--tools",
                "",
                # Disable user-level customizations such as CLAUDE.md, skills, hooks, plugins, and MCP;
                # auth and model selection are unaffected (do not use --bare; it disables OAuth).
                "--safe-mode",
            ]
            # When the model name is blank, keep the CLI's own default model so a model ID hardcoded here
            # does not go stale as the models available on the subscription change.
            if model_name:
                command += ["--model", model_name]

            cli_env, removed_env = build_claude_code_env()
            if removed_env:
                # Log variable names only, not values, so secrets are not written into the log.
                logger.warning(
                    f"{llm_provider}: ignoring conflicting environment variables "
                    f"so the subscription login is used: {', '.join(removed_env)}"
                )

            logger.info(f"invoking claude cli, model: {model_name or 'cli default'}")
            # The CLI reads CLAUDE.md and project settings from the working directory, which would contaminate
            # the copy, so it always runs inside a temporary empty directory.
            with tempfile.TemporaryDirectory() as work_dir:
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        # The CLI always emits UTF-8. Without an explicit encoding,
                        # text=True decodes with the system locale (e.g. cp1252 on
                        # non-English Windows), so every non-ASCII character reaches
                        # the script as mojibake.
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_seconds,
                        cwd=work_dir,
                        env=cli_env,
                    )
                except subprocess.TimeoutExpired:
                    raise Exception(
                        f"[{llm_provider}] claude cli timed out after "
                        f"{timeout_seconds:.0f}s"
                    )

            # Failures such as not logged in or quota exhausted also return JSON (`is_error` is true and
            # `result` is a readable reason); only the exit code is non-zero. Parse stdout first,
            # and fall back to the exit code and stderr only when JSON is not available.
            stdout = (completed.stdout or "").strip()
            try:
                payload = json.loads(stdout) if stdout else None
            except json.JSONDecodeError:
                payload = None

            if payload is None:
                detail = (completed.stderr or stdout or "").strip()
                if "unknown option" in detail.lower():
                    raise Exception(
                        f"[{llm_provider}] the installed claude CLI does not support "
                        f"the required isolation flags; upgrade to "
                        f"{CLAUDE_CODE_MIN_CLI_VERSION} or newer: {detail[:300]}"
                    )
                if completed.returncode != 0:
                    raise Exception(
                        f"[{llm_provider}] claude cli exited with code "
                        f"{completed.returncode}: {detail[:500]}"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{detail[:500]}"'
                )

            if payload.get("is_error") or completed.returncode != 0:
                reason = str(payload.get("result") or "").strip() or (
                    f"claude cli exited with code {completed.returncode}"
                )
                # An interactive /login cannot run in a container, so give the auth methods that work here.
                if "login" in reason.lower():
                    reason += (
                        " (run `claude setup-token` on the host and pass the token "
                        "to the container as CLAUDE_CODE_OAUTH_TOKEN)"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an error response: "{reason[:500]}"'
                )

            return _normalize_text_response(payload.get("result"), llm_provider)

        if adapter == "kimi_code":
            # A Kimi subscription (the Kimi Code benefit on Plus+) does not issue a metered API key. Auth is
            # done by OAuth device authorization: the WebUI has the user confirm once in the browser and stores
            # a refresh token locally; the access token expires in 15 minutes and kimi_code_oauth renews it
            # transparently. Requests use the official OpenAI-compatible endpoint, a quota separate from the metered API.
            from app.services import kimi_code_oauth

            try:
                timeout_seconds = kimi_code_oauth.coerce_timeout(
                    extra_values.get("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            access_token = kimi_code_oauth.get_valid_access_token()
            # The authorized region follows the credential: a token is valid only in the region that issued it. After the user switches
            # API Platform, a request sent to the UI region would send a China-issued token to the
            # Global endpoint (always 401), so the request URL follows the token's issuing region first.
            authorized_region = kimi_code_oauth.authorized_region()
            if authorized_region:
                request_base_url = kimi_code_oauth.api_base_url(authorized_region)
                if base_url and authorized_region != kimi_code_oauth.resolve_region(
                    base_url
                ):
                    logger.warning(
                        f"kimi_code: API Platform selection differs from the "
                        f"authorized region; using '{authorized_region}' to match "
                        "the issued token"
                    )
            else:
                request_base_url = base_url or kimi_code_oauth.api_base_url(
                    kimi_code_oauth.resolve_region(base_url)
                )
            # When the model name is blank, default to kimi-for-coding, the dedicated coding model in the subscription,
            # so a hardcoded general model id does not go stale as the benefit changes.
            request_model = model_name or "kimi-for-coding"
            logger.info(f"requesting kimi code chat completion, model: {request_model}")
            client = sdk_client = OpenAI(
                api_key=access_token,
                base_url=request_base_url,
                timeout=timeout_seconds,
                # The server anti-bot pipeline requires the same device-identity headers as the CLI.
                default_headers=kimi_code_oauth.api_headers(),
            )
            response = client.chat.completions.create(
                model=request_model, messages=[{"role": "user", "content": prompt}]
            )
            if response and isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            raise Exception(
                f'[{llm_provider}] returned an invalid response: "{response}"'
            )

        if adapter == "modelscope":
            content = ""
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            response = sdk_stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
        }
        if llm_provider == "openrouter":
            client_kwargs["timeout"] = _OPENROUTER_TIMEOUT_SECONDS
            client_kwargs["default_headers"] = {
                "X-Title": "Script-to-Reality Renderer Studio",
            }
        client = sdk_client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        message = _sanitize_error_message(e)
        if llm_provider == "openrouter":
            message = _classify_openrouter_error(e, message)
        return f"Error: {message}"

    finally:
        for resource in (sdk_stream, sdk_client):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as cleanup_error:
                    logger.warning(
                        f"could not close LLM transport: {type(cleanup_error).__name__}"
                    )


def test_connection() -> tuple[bool, str, float]:
    """
    Send one minimal request with the current Provider config to check that generation actually works.

    The connection test reuses `_generate_response()`, so it covers the API key, base URL,
    model name, and Provider-specific fields, but it does not enter script-generation retries and does not send
    the user's video subject or copy. The return values are success, error text, and elapsed time.
    """
    started_at = perf_counter()
    response = _generate_response(prompt="Reply with exactly: OK")
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = response.removeprefix("Error:").strip()
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # The API layer already checks length with Pydantic; this extra guard is so
    # a direct generate_script call from the WebUI or an internal service does not send an overlong prompt to the model,
    # which would spike token cost or fail the request.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # The WebUI and the API both clamp the range; this guards internal calls so bad parameters do not directly inflate
        # LLM generation cost or produce an empty result.
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # Join "script generation rules" and "runtime context" separately. Even if an advanced user overrides the default
    # system prompt, the video subject, language, and paragraph count required on every generation are not dropped.
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
Apply every rule above to this job: factual accuracy, the opening hook, narrative structure, retention, spoken delivery, pacing, visual clarity, language, the silent quality check, and the output contract. Do not invent facts to fill length.
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax.  Use non-greedy .*? so each bracket/paren
        # group is removed independently; the greedy form would eat all text
        # between the first opener and the last closer on the same line.
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt=prompt)
            else:
                response = _generate_response(prompt=prompt, app_config=app_config)
            if isinstance(response, str) and response.startswith("Error: "):
                # _generate_response returns provider failures as text. Passing
                # that text through would make the task treat it as narration.
                raise ValueError(response)
            if response:
                candidate = format_response(response)
            else:
                logging.error("gpt returned an empty response")
                candidate = ""

            # Some upstream providers may return quota errors as plain text.
            if candidate and "当日额度已消耗完" in candidate:
                raise ValueError(candidate)

            if candidate:
                final_script = candidate
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries - 1:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if not final_script:
        logger.error("failed to generate video script after retries")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    video_script = utils.remove_pause_tags(video_script or "").strip()
    if match_script_order:
        goal = (
            f"Generate {amount} chronological stock-video search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        # In ordered-keyword mode the example count must match amount, so a fixed
        # set of 4 examples does not mislead the model into returning too few keywords for long copy and hurting footage coverage.
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Footage Search Terms Writer

You name pictures a camera can actually film. Use only subjects the script mentions. Do not invent people, places, or events, and do not turn a slogan or a caption into a search term.

## Goals:
{goal}

## Constraints:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words and name a concrete visible subject. Always include the main subject of the video when that moment shows it.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video and to the moment they describe.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        search_terms = []
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                # generate_terms's public return type is List[str]. If the Provider
                # error text is returned as-is, a downstream empty check treats a non-empty string as success,
                # and the footage download loop also walks that error text character by character, making pointless external requests.
                # Always return an empty list so the task orchestrator stops the task at the real failure.
                logger.error(f"failed to generate video terms: {response}")
                return []
            search_terms = json.loads(_strip_code_fence(response))
        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # Keep the retry path, but the non-standard JSON from the LLM must be logged,
                        # otherwise a later look at empty search terms cannot tell
                        # a model-format problem from a parse-logic problem.
                        logger.warning(f"failed to generate video terms: {str(e)}")

        # Apply the same contract to direct JSON and prose-wrapped recovery.
        # Otherwise a nonempty array of numbers or objects reaches material search.
        if not isinstance(search_terms, list) or not all(
            isinstance(term, str) for term in search_terms
        ):
            logger.error("response is not a list of strings.")
            search_terms = []

        # The model may ignore the requested count or return blank strings.
        # Enforce the prompt contract before material providers are contacted;
        # repeated topics remain meaningful in chronological mode.
        search_terms = [term.strip() for term in search_terms if term.strip()][:amount]
        if search_terms:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# Social publishing metadata
#
# From the video subject and script, generate the title, caption, and hashtags commonly used when publishing to short-video platforms.
# This only reuses the existing LLM provider. It does not call any external publishing service and does not affect the main video pipeline.
# =============================================================================

# Platforms differ in preferred copy length and hashtag count. A conservative cap is used so that after the model returns
# overly long content the caller does not have to trim it again.
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# Generic fallback tags used when the LLM is unavailable. They are deliberately not tied to one country or language, so the API
# can still return a usable structure for Chinese, English, Vietnamese, and other cases.
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # The API layer limits length; this extra guard protects internal calls, or a future WebUI
    # calling this directly, from sending overlong content to the model and spiking token cost.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    Normalize hashtags returned by the LLM into `#tag` form.

    The LLM may return a string, an array, a phrase with spaces, duplicate tags, or punctuation.
    Cleaning them in one place keeps the response shape stable and avoids empty tags,
    duplicate tags, or hashtags that do not match the usual format when publishing.
    """
    if isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    elif isinstance(raw, (list, tuple)):
        # Each array item is treated as one complete tag, so "du lich" becomes
        # "#dulich" rather than two tags.
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        tag = re.sub(r"[^\w]", "", item, flags=re.UNICODE)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Factual Short-Video Publisher

You write publishing metadata that matches the narration. Specific and accurate beats catchy. Do not invent facts, statistics, or events that are not in the subject or script. Do not use clickbait, fake urgency, or generic AI phrasing.

## Goal
Write publishing metadata for a short video that will be posted on {label}. The title and caption must describe what the narration actually says.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a specific, truthful hook taken from the subject, at most {spec["title_max"]} characters. Do not use "you won't believe", "shocking", or "this changes everything".
4. "caption": a clear description of the real subject, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption. Do not ask for likes, comments, shares, or follows.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and name the actual topic for {label}. Do not pad the list with unrelated trend tags.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # Some models wrap the JSON in explanatory text or a markdown fence.
        # API callers only need a stable structure, so try to extract the first JSON object.
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if match:
            data = json.loads(match.group())

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # When there is no subject, use the script's first sentence as a title fallback so the API does not return an empty title.
        title = re.split(r"(?<=[.!?。！？])\s+", script)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    Generate short-video publishing copy metadata.

    The return shape is fixed as `{"title": str, "caption": str, "hashtags": List[str]}`.
    If the LLM is unavailable or the format is invalid, this falls back to a generic heuristic so API
    callers always receive a structure they can display and edit before publishing.
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and response.startswith("Error: "):
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)

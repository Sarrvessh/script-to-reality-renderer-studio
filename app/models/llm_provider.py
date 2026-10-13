from dataclasses import dataclass


DEFAULT_LLM_PROVIDER_ID = "moonshot"


@dataclass(frozen=True, slots=True)
class LLMProviderField:
    """Describes extra Provider settings beyond the API key, base URL, and model name."""

    config_suffix: str
    label_key: str
    required: bool = False
    secret: bool = False
    default_value: str = ""


@dataclass(frozen=True, slots=True)
class LLMProviderEndpoint:
    """Describes the matching entry points and API addresses a Provider uses in each service region."""

    endpoint_id: str
    default_label: str
    base_url: str
    api_key_url: str
    model_docs_url: str = ""


@dataclass(frozen=True, slots=True)
class LLMProviderSpec:
    """
    Central declaration of an LLM Provider.

    This stores the stable metadata shared by the WebUI, config loading, and service calls, including the default
    display name and locale key. It does not store translated copy, and it does not implement API requests. In this
    split, the Registry owns what a Provider is, and the service-layer adapter still owns how it is called.
    """

    provider_id: str
    default_label: str
    adapter: str = "openai_compatible"
    api_key_url: str = ""
    default_model: str = ""
    default_base_url: str = ""
    model_docs_url: str = ""
    requires_api_key: bool = True
    requires_model_name: bool = True
    requires_base_url: bool = True
    show_api_key: bool = True
    show_base_url: bool = True
    deprecated_models: tuple[str, ...] = ()
    deprecated_base_urls: tuple[str, ...] = ()
    extra_fields: tuple[LLMProviderField, ...] = ()
    service_endpoints: tuple[LLMProviderEndpoint, ...] = ()
    default_service_endpoint_id: str = ""
    international_service_endpoint_id: str = ""

    @property
    def label_key(self) -> str:
        return f"llm_provider_label.{self.provider_id}"

    @property
    def tips_key(self) -> str:
        return f"llm_provider_tips.{self.provider_id}"

    @property
    def endpoint_selector_label_key(self) -> str:
        return f"llm_provider_endpoint_selector.{self.provider_id}"

    @property
    def endpoint_selector_help_key(self) -> str:
        return f"llm_provider_endpoint_selector_help.{self.provider_id}"

    @property
    def authentication_error_key(self) -> str:
        return f"llm_provider_authentication_error.{self.provider_id}"

    def endpoint_label_key(self, endpoint_id: str) -> str:
        return f"llm_provider_endpoint.{self.provider_id}.{endpoint_id}"

    def config_key(self, suffix: str) -> str:
        return f"{self.provider_id}_{suffix}"

    def resolve_model_name(self, configured_model: str | None) -> str:
        """Resolve an empty value or a retired historical default to the current default model."""
        model_name = (configured_model or "").strip()
        if not model_name or model_name in self.deprecated_models:
            return self.default_model
        return model_name

    def resolve_base_url(self, configured_base_url: str | None) -> str:
        """Resolve the base URL, and migrate a retired historical address to the current default."""
        base_url = (configured_base_url or "").strip()
        deprecated_urls = {url.rstrip("/") for url in self.deprecated_base_urls}
        if not base_url or base_url.rstrip("/") in deprecated_urls:
            return self.effective_default_base_url
        return base_url

    def get_service_endpoint(self, endpoint_id: str) -> LLMProviderEndpoint | None:
        """Look up a service region by its stable ID, so business logic does not depend on changeable promo links."""
        return next(
            (
                endpoint
                for endpoint in self.service_endpoints
                if endpoint.endpoint_id == endpoint_id
            ),
            None,
        )

    @property
    def default_service_endpoint(self) -> LLMProviderEndpoint | None:
        """Return the default service region declared by the Provider."""
        return self.get_service_endpoint(self.default_service_endpoint_id)

    @property
    def international_service_endpoint(self) -> LLMProviderEndpoint | None:
        """Return the international service region declared by the Provider."""
        return self.get_service_endpoint(self.international_service_endpoint_id)

    @property
    def effective_default_base_url(self) -> str:
        """Prefer the base URL from the default service region; ordinary Providers still use the original field."""
        endpoint = self.default_service_endpoint
        return endpoint.base_url if endpoint else self.default_base_url

    def preferred_service_endpoint(
        self, *, prefer_international: bool
    ) -> LLMProviderEndpoint | None:
        """Return the preferred entry for the UI region, falling back to the default when no international entry exists."""
        if prefer_international and self.international_service_endpoint:
            return self.international_service_endpoint
        return self.default_service_endpoint

    def effective_api_key_url(self, *, prefer_international: bool = False) -> str:
        """Resolve the API key signup entry in one place, so endpoint Providers do not maintain the link twice."""
        endpoint = self.preferred_service_endpoint(
            prefer_international=prefer_international
        )
        return endpoint.api_key_url if endpoint else self.api_key_url

    def effective_model_docs_url(self, *, prefer_international: bool = False) -> str:
        """Resolve the model list and docs entry in one place, so endpoint Providers do not maintain the link twice."""
        endpoint = self.preferred_service_endpoint(
            prefer_international=prefer_international
        )
        return endpoint.model_docs_url if endpoint and endpoint.model_docs_url else self.model_docs_url

    def find_service_endpoint(
        self, configured_base_url: str | None
    ) -> LLMProviderEndpoint | None:
        """Identify the Provider's standard service region from the saved base URL."""
        normalized_url = (configured_base_url or "").strip().rstrip("/")
        if not normalized_url:
            return None
        return next(
            (
                endpoint
                for endpoint in self.service_endpoints
                if endpoint.base_url.rstrip("/") == normalized_url
            ),
            None,
        )

    def select_service_endpoint(
        self,
        configured_base_url: str | None,
        *,
        has_api_key: bool,
        prefer_international: bool,
    ) -> LLMProviderEndpoint | None:
        """
        Choose the standard service region the WebUI should show.

        A saved standard address wins; an unknown address stays custom. Older configs may have only
        an API key and no base URL, and those users keep the Registry default region so an
        upgrade does not switch services just because the UI language differs. Only a brand-new config picks the
        international entry from the UI language.
        """
        configured_url = (configured_base_url or "").strip()
        if configured_url:
            return self.find_service_endpoint(configured_url)

        default_endpoint = self.default_service_endpoint
        if has_api_key or not prefer_international:
            return default_endpoint

        return self.preferred_service_endpoint(
            prefer_international=prefer_international
        )


# Tuple order is the WebUI dropdown order. To add an ordinary OpenAI-compatible Provider,
# add one entry here and a locale; only a Provider with a different protocol needs a matching adapter
# in app/services/llm.py.
LLM_PROVIDER_REGISTRY = (
    # Recommended Providers
    LLMProviderSpec(
        "moonshot",
        "Kimi / Moonshot AI",
        default_model="kimi-k3",
        service_endpoints=(
            LLMProviderEndpoint(
                endpoint_id="china",
                default_label="China",
                base_url="https://api.moonshot.cn/v1",
                api_key_url=(
                    "https://platform.kimi.com?"
                    "track_id=track-6eec1e56a4494e52adcaebbcbbefce59&"
                    "aff=script-to-reality"
                ),
                model_docs_url=(
                    "https://platform.kimi.com/docs/models?"
                    "track_id=track-6eec1e56a4494e52adcaebbcbbefce59&"
                    "aff=script-to-reality"
                ),
            ),
            LLMProviderEndpoint(
                endpoint_id="global",
                default_label="Global",
                base_url="https://api.moonshot.ai/v1",
                api_key_url=(
                    "https://platform.kimi.ai?"
                    "track_id=track-9e3b711aa2594e378f6fe5b8de718a76&"
                    "aff=script-to-reality"
                ),
                model_docs_url=(
                    "https://platform.kimi.ai/docs/models?"
                    "track_id=track-9e3b711aa2594e378f6fe5b8de718a76&"
                    "aff=script-to-reality"
                ),
            ),
        ),
        default_service_endpoint_id="china",
        international_service_endpoint_id="global",
    ),
    # The Kimi subscription (Kimi Code on Plus and above) uses OAuth device authorization:
    # after the user confirms once in the browser, the service layer calls the official
    # OpenAI-compatible endpoint with a Bearer token, and no pay-as-you-go API key is required. The China site and Global use separate
    # accounts and authorization systems; each endpoint carries its auth page, API address, and subscription link.
    LLMProviderSpec(
        "kimi_code",
        "Kimi Code (Kimi subscription)",
        adapter="kimi_code",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        requires_model_name=False,
        default_model="kimi-for-coding",
        extra_fields=(
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
        service_endpoints=(
            LLMProviderEndpoint(
                endpoint_id="china",
                default_label="China",
                base_url="https://api.kimi.com/coding/v1",
                api_key_url="https://www.kimi.com/code?aff=script-to-reality",
                model_docs_url="https://platform.kimi.com/docs/models?aff=script-to-reality",
            ),
            LLMProviderEndpoint(
                endpoint_id="global",
                default_label="Global",
                base_url="https://api.kimi.ai/coding/v1",
                api_key_url="https://www.kimi.ai/code?aff=script-to-reality",
                model_docs_url="https://platform.kimi.ai/docs/models?aff=script-to-reality",
            ),
        ),
        default_service_endpoint_id="china",
        international_service_endpoint_id="global",
    ),
    # First-party model vendors and cloud providers
    LLMProviderSpec(
        "openai",
        "OpenAI",
        api_key_url="https://platform.openai.com/api-keys",
        default_model="gpt-5.5",
        default_base_url="https://api.openai.com/v1",
    ),
    LLMProviderSpec(
        "anthropic",
        "Anthropic Claude",
        api_key_url="https://platform.claude.com/settings/keys",
        default_model="claude-sonnet-5",
        default_base_url="https://api.anthropic.com/v1/",
    ),
    LLMProviderSpec(
        "gemini",
        "Google Gemini",
        adapter="gemini",
        api_key_url="https://aistudio.google.com/app/apikey",
        default_model="gemini-3.1-pro-preview",
        requires_base_url=False,
        show_base_url=False,
        deprecated_models=("gemini-pro", "gemini-1.0-pro"),
    ),
    LLMProviderSpec(
        "deepseek",
        "DeepSeek",
        api_key_url="https://platform.deepseek.com/api_keys",
        default_model="deepseek-v4-pro",
        default_base_url="https://api.deepseek.com",
    ),
    LLMProviderSpec(
        "qwen",
        "Alibaba Cloud Qwen",
        adapter="qwen",
        api_key_url="https://dashscope.console.aliyun.com/apiKey",
        default_model="qwen-max",
        requires_base_url=False,
        show_base_url=False,
    ),
    LLMProviderSpec(
        "azure",
        "Microsoft Azure OpenAI",
        adapter="azure",
        api_key_url=(
            "https://portal.azure.com/#view/"
            "Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI"
        ),
        default_model="gpt-35-turbo",
    ),
    LLMProviderSpec(
        "volcengine",
        "ByteDance VolcEngine Ark",
        api_key_url=(
            "https://www.volcengine.com/activity/ai618?utm_campaign=hw&"
            "utm_content=hw&utm_medium=devrel_tool_web&utm_source=OWO&"
            "utm_term=Script-to-Reality Renderer Studio"
        ),
        default_model="doubao-seed-2-1-turbo-260628",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    LLMProviderSpec(
        "grok",
        "xAI Grok",
        api_key_url="https://console.x.ai/",
        default_model="grok-4.3",
        default_base_url="https://api.x.ai/v1",
    ),
    LLMProviderSpec(
        "minimax",
        "MiniMax",
        api_key_url="https://platform.minimax.io/",
        default_model="MiniMax-M3",
        default_base_url="https://api.minimax.io/v1",
    ),
    LLMProviderSpec(
        "mimo",
        "Xiaomi MiMo",
        api_key_url=(
            "https://platform.xiaomimimo.com/docs/zh-CN/quick-start/first-api-call"
        ),
        default_model="mimo-v2.5-pro",
        default_base_url="https://api.xiaomimimo.com/v1",
    ),
    # iFlytek Astron MaaS pay-as-you-go and Token Plan use different OpenAI-compatible addresses,
    # and their API keys cannot be mixed. The default here is pay-as-you-go; Token Plan users set the Base URL
    # to the maas-token-api address.
    LLMProviderSpec(
        "iflytek",
        "iFlytek Spark (Astron MaaS)",
        api_key_url="https://maas.xfyun.cn/",
        default_model="spark-x2.5",
        default_base_url="https://maas-api.cn-huabei-1.xf-yun.com/v2",
        model_docs_url="https://maas.xfyun.cn/modelSquare",
    ),
    # Aggregators and unified access platforms
    LLMProviderSpec(
        "shengsuanyun",
        "Shengsuan Cloud",
        api_key_url="https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
        default_model="deepseek/deepseek-v4-flash",
        default_base_url="https://router.shengsuanyun.com/api/v1",
    ),
    # APIMart offers both an `/api/v1` business API and a `/v1` OpenAI-compatible API.
    # The current LLM service layer uses the OpenAI SDK to read choices directly, so it must use the
    # `/v1` entry without a code/data wrapper, and cannot copy the async business API address.
    LLMProviderSpec(
        "apimart",
        "APIMart",
        api_key_url="https://go.apimart.ai/gh-script-to-reality",
        default_model="gpt-5.6-terra",
        default_base_url="https://api.apimart.ai/v1",
    ),
    LLMProviderSpec(
        "cloudflare",
        "Cloudflare AI Gateway",
        adapter="cloudflare_ai_gateway",
        api_key_url="https://dash.cloudflare.com/",
        default_model="openai/gpt-4.1-mini",
        requires_base_url=False,
        show_base_url=False,
        deprecated_models=("@cf/meta/llama-3.1-8b-instruct",),
        extra_fields=(
            LLMProviderField("account_id", "Account ID", required=True),
            LLMProviderField(
                "gateway_id",
                "Gateway ID",
                default_value="default",
            ),
        ),
    ),
    LLMProviderSpec(
        "modelscope",
        "Alibaba ModelScope",
        adapter="modelscope",
        api_key_url=("https://modelscope.cn/docs/model-service/API-Inference/intro"),
        default_model="ZhipuAI/GLM-5.2",
        default_base_url="https://api-inference.modelscope.cn/v1/",
    ),
    LLMProviderSpec(
        "aihubmix",
        "AIHubMix",
        api_key_url="https://aihubmix.com/",
        default_model="gpt-5.4-mini",
        default_base_url="https://aihubmix.com/v1",
    ),
    LLMProviderSpec(
        "aimlapi",
        "AIML API",
        api_key_url="https://aimlapi.com/app/keys",
        default_model="openai/gpt-5-5",
        default_base_url="https://api.aimlapi.com/v1",
    ),
    LLMProviderSpec(
        "evolink",
        "EvoLink",
        api_key_url="https://evolink.ai/dashboard/keys",
        default_model="gpt-5.5",
        default_base_url="https://direct.evolink.ai/v1",
    ),
    LLMProviderSpec(
        "openrouter",
        "OpenRouter",
        api_key_url="https://openrouter.ai/settings/keys",
        # OpenRouter's own free router. Specific ":free" model IDs are removed
        # often, so the default stays on this stable alias instead of a vendor
        # model that can 404. Set openrouter_model_name to override it.
        default_model="openrouter/free",
        default_base_url="https://openrouter.ai/api/v1",
        deprecated_models=(
            "minimax/minimax-m3:free",
            "z-ai/glm-5.2:free",
        ),
    ),
    LLMProviderSpec(
        "api_route",
        "API Route",
        api_key_url="https://www.api-route.com",
        default_model="gpt-5.4-mini",
        default_base_url="https://www.api-route.com/v1",
        model_docs_url="https://www.api-route.com/pricing",
    ),
    # Fluxion's OpenAI group uses the /v1 API and reuses the existing Chat Completions adapter.
    # The group decides the available models and address, so this only supplies defaults and still lets users override them;
    # native Anthropic/Gemini groups cannot use this entry, which avoids a protocol mismatch.
    LLMProviderSpec(
        "fluxionai",
        "Fluxion AI",
        api_key_url=(
            "https://fluxionai.space/register?source=github"
            "&campaign=script-to-reality&promo=STUDIO"
        ),
        default_model="gpt-5.5",
        default_base_url="https://fluxionai.space/v1",
        model_docs_url="https://fluxionai.space/model-plaza",
    ),
    LLMProviderSpec(
        "cheaperinference",
        "Cheaper Inference",
        api_key_url="https://cheaperinference.com/signup",
        default_model="gpt-5.4-mini",
        default_base_url="https://api.cheaperinference.com/v1",
        model_docs_url="https://cheaperinference.com/#models",
    ),
    LLMProviderSpec(
        "requesty",
        "Requesty",
        api_key_url="https://app.requesty.ai/api-keys",
        default_model="openai/gpt-5.4-mini",
        default_base_url="https://router.requesty.ai/v1",
        model_docs_url="https://www.requesty.ai/models",
    ),
    LLMProviderSpec(
        "futureinfra",
        "FutureInfra",
        api_key_url="https://futureinfra.ai/console/?screen=ai-router",
        default_model="openai/gpt-4o-mini",
        default_base_url="https://futureinfra.ai/v1/ai",
        model_docs_url="https://futureinfra.ai/ai/",
    ),
    LLMProviderSpec(
        "yapi",
        "Y-API",
        api_key_url="https://y-api.bestvirtualgoods.com/app/keys",
        default_model="deepseek/deepseek-v4-flash",
        default_base_url="https://api.y-api.bestvirtualgoods.com/v1",
        model_docs_url="https://y-api.bestvirtualgoods.com/models",
    ),
    LLMProviderSpec(
        "opper",
        "Opper",
        api_key_url="https://platform.opper.ai",
        default_model="gpt-5.4-mini",
        default_base_url="https://api.opper.ai/v3/compat",
        model_docs_url="https://opper.ai/models",
    ),
    LLMProviderSpec(
        "onomeo",
        "onomeo",
        api_key_url="https://onomeo.com/dashboard",
        default_model="auto",
        default_base_url="https://onomeo.com/v1",
        model_docs_url="https://onomeo.com/models",
    ),
    # Local deployments and generic gateways
    LLMProviderSpec(
        "ollama",
        "Ollama",
        requires_api_key=False,
        show_api_key=False,
    ),
    # A Claude subscription (Pro / Max / Team) does not issue an API key; only the official Claude Code
    # client can use the credential, so this Provider does not call an HTTP API and instead invokes the locally signed-in
    # claude CLI. An empty model name keeps the CLI's current default model.
    LLMProviderSpec(
        "claude_code",
        "Claude Code (Claude subscription)",
        adapter="claude_code",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        requires_model_name=False,
        extra_fields=(
            LLMProviderField("cli_path", "Claude CLI Path"),
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
    ),
    LLMProviderSpec(
        "oneapi",
        "OneAPI",
        api_key_url="https://github.com/songquanpeng/one-api",
    ),
    LLMProviderSpec(
        "litellm",
        "LiteLLM",
        adapter="litellm",
        default_model="openai/gpt-4o-mini",
        requires_api_key=False,
        requires_base_url=False,
        show_api_key=False,
        show_base_url=False,
    ),
    # Other inference and public services
    LLMProviderSpec(
        "groq",
        "Groq",
        api_key_url="https://console.groq.com/keys",
        default_model="openai/gpt-oss-120b",
        default_base_url="https://api.groq.com/openai/v1",
    ),
    LLMProviderSpec(
        "pollinations",
        "Pollinations AI",
        api_key_url="https://enter.pollinations.ai/",
        default_model="openai-fast",
        default_base_url="https://gen.pollinations.ai/v1",
        deprecated_models=("default",),
        deprecated_base_urls=("https://text.pollinations.ai/openai",),
    ),
)

LLM_PROVIDERS = {provider.provider_id: provider for provider in LLM_PROVIDER_REGISTRY}

if len(LLM_PROVIDERS) != len(LLM_PROVIDER_REGISTRY):
    raise RuntimeError("duplicate LLM provider id in registry")


def get_llm_provider(provider_id: str) -> LLMProviderSpec | None:
    return LLM_PROVIDERS.get((provider_id or "").lower())


def normalize_provider_override(value: str | None, default_value: str | None) -> str:
    """
    Keep only user overrides that differ from the Registry default.

    The WebUI needs to show the default in the input, but that must not freeze the default into config.toml;
    otherwise, after the Registry default model or address is upgraded, the old config would keep overriding the new default.
    """
    normalized_value = (value or "").strip()
    normalized_default = (default_value or "").strip()
    if normalized_value == normalized_default:
        return ""
    return normalized_value

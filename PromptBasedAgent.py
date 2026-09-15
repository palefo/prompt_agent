import os
import json
import datetime
from typing import Any, TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt


PROMPT_NAME = "agent.prompt"
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", PROMPT_NAME)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
USE_AZURE = os.environ.get("USE_AZURE", "false").strip().lower() in ("1", "true", "yes", "on")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))

class AgentContext(TypedDict, total=False):
    choices: dict[str, Any]


def get_current_date() -> str:
    """Get today's date in ISO format."""
    return datetime.date.today().isoformat()

def _load_system_prompt(choices: dict | None = None) -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read().strip()

    # Load choices and substitute simple {{key}} placeholders
    choices = choices or {}
    if isinstance(choices, dict):
        for k, v in choices.items():
            if isinstance(v, (str, int, float)):
                template = template.replace("{{" + k + "}}", str(v))

    # Also append the raw choices JSON so the model can reference all values
    if choices:
        template += "\n\n# INPUT DATA\n" + json.dumps(choices, indent=2, ensure_ascii=False)

    return template

@dynamic_prompt
def current_system_prompt(request: ModelRequest[AgentContext]) -> str:
    context = request.runtime.context or {}
    return _load_system_prompt(context.get("choices"))


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set (see .env.example)")
    return value


def _build_model():
    """Azure OpenAI when USE_AZURE=true, otherwise the OpenAI API (OPENAI_API_KEY)."""
    if USE_AZURE:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_require_env("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", OPENAI_MODEL),
            timeout=LLM_TIMEOUT,
            max_retries=2,
        )
    from langchain_openai import ChatOpenAI
    _require_env("OPENAI_API_KEY")
    return ChatOpenAI(model=OPENAI_MODEL, timeout=LLM_TIMEOUT, max_retries=2)


graph = create_agent(
    model=_build_model(),
    tools=[get_current_date],
    middleware=[current_system_prompt],
    context_schema=AgentContext,
)

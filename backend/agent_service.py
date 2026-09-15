import logging
import os
import re
import uuid
import json
from typing import Any
from functools import lru_cache

log = logging.getLogger(__name__)

# ── 1. Load configuration before importing the agent ────────────────────────

def _bootstrap_env() -> None:
    """Load local environment variables for development."""

    # .env file (local development)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

_bootstrap_env()

# ── 2. Import agent (env vars must be set first) ─────────────────────────────

@lru_cache(maxsize=1)
def _get_agent_components():
    from PromptBasedAgent import graph  # noqa: E402
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
    return graph, AIMessage, HumanMessage


def warmup_agent() -> None:
    """Initialize the agent once so later requests are faster."""
    graph, AIMessage, HumanMessage = _get_agent_components()
    # add choices to history


class AgentUnavailable(RuntimeError):
    """The language model could not produce a reply (network, quota, filter...)."""

# ── 3. Helpers ────────────────────────────────────────────────────────────────

def make_thread_id(seed: str) -> str:
    """Return a deterministic UUID5 from the session seed."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def _extract_json_object(text: str):
    """Find the first balanced JSON object in *text*, regardless of language."""
    for brace_idx, character in enumerate(text):
        if character != "{":
            continue

        depth = 0
        in_string = False
        escaped = False
        for index in range(brace_idx, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[brace_idx:index + 1])
                    except json.JSONDecodeError:
                        break
                    # Drop a Markdown code fence opened just before the JSON.
                    leading = re.sub(r"```(?:json)?\s*$", "", text[:brace_idx].rstrip())
                    return leading.strip(), parsed

    return text, None


def _message_text(message) -> str:
    content = message.content if hasattr(message, "content") else message.get("content", message)
    if isinstance(content, list):  # content blocks
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def generate_reply(
    history: list[dict[str, str]],
    thread_id: str,
    choices: dict[str, Any] | None = None,
) -> tuple[str, dict | None]:
    """Invoke the LangGraph agent and return (text, parsed_json).

    If the agent outputs a thank-you marker followed by a JSON object, the JSON
    will be parsed and returned separately. Otherwise parsed_json is None.
    Raises AgentUnavailable when the model call fails; details go to the log,
    never to the participant.
    """

    graph, AIMessage, HumanMessage = _get_agent_components()

    # Build LangChain message list for the graph
    lc_messages = []
    for message in history:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        result = graph.invoke(
            {"messages": lc_messages},
            config=config,
            context={"choices": choices or {}},
        )
        content = _message_text(result["messages"][-1]).strip()
    except Exception as exc:
        log.exception("Agent call failed for thread %s", thread_id)
        raise AgentUnavailable(str(exc)) from exc

    # Completion text may be translated, so detect the JSON object itself.
    leading, parsed = _extract_json_object(content)
    if parsed is not None:
        return leading, parsed
    if not content:
        raise AgentUnavailable("The model returned an empty reply")
    return content, None

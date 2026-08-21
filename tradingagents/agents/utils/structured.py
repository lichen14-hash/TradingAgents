"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.

    Returns markdown only. Callers that need to know *whether* the fallback
    fired — the three decision agents, whose typed rating is lost when it does
    — should use :func:`invoke_structured_guarded` instead.
    """
    result, _parsed, _findings = invoke_structured_guarded(
        structured_llm, plain_llm, prompt, render, agent_name, section="",
    )
    return result


def invoke_structured_guarded(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    section: str,
) -> tuple[str, T | None, list[dict[str, str]]]:
    """Same fallback behaviour, but report what was lost.

    Returns ``(markdown, parsed_or_None, findings)``:

    - ``parsed`` is the typed Pydantic instance, or ``None`` when the run went
      through the free-text path. A caller that needs a machine-readable field
      (e.g. the Research Manager's ``recommendation``) must treat ``None`` as
      "parse it out of the prose, and flag that you had to".
    - ``findings`` carries a warn-level integrity finding when the fallback
      fired, so the degradation reaches the report instead of only the log.

    Why this exists: in the 2026-08-11 300760.SZ incident both runs' Research
    Manager silently degraded to free text. The rendered reports contained no
    ``**Recommendation**`` line at all, so nothing downstream could tell that
    the Portfolio Manager had overridden a research view — the view was simply
    not recorded. A single ``logger.warning`` was not enough.
    """
    from tradingagents.agents.utils.integrity import (
        invoke_text_guarded,
        structured_degradation_finding,
    )

    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result), result, []
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )
            reason: Exception | str = exc
    else:
        reason = "provider does not support with_structured_output"

    # The free-text fallback gets the same empty-output guard as the debate
    # agents: a degraded call that also comes back blank must not leave the
    # section silently empty.
    text, blank_findings = invoke_text_guarded(
        plain_llm, prompt, section=section or agent_name, role=agent_name,
    )
    findings = [structured_degradation_finding(section or agent_name, agent_name, reason)]
    findings.extend(blank_findings)
    return text, None, findings

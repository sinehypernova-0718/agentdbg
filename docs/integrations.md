# Integrations

## Philosophy

AgentDbg is **framework-agnostic** at the core. The SDK is a thin layer: you call `@trace` and `record_llm_call` / `record_tool_call` / `record_state` from any Python code. No required dependency on LangChain, OpenAI Agents SDK, or others.

**Adapters** are thin translation layers: they hook into a framework's callbacks and emit AgentDbg events. They do not lock you into that framework for the rest of your app.

---

## Available in v0.1

### LangChain / LangGraph callback handler

**Status: available.** An optional callback handler lives at `agentdbg.integrations.langchain`. It records LLM calls and tool calls to the active AgentDbg run automatically.

**Requirements:** `langchain-core` must be installed. Install the optional dependency group:

```bash
pip install -e ".[langchain]"
```

If `langchain-core` is not installed, importing the integration raises a clear `ImportError` with install instructions. The integration is optional; the core package does not depend on it.

**Usage:**

```python
from agentdbg import trace
from agentdbg.integrations import AgentDbgLangChainCallbackHandler

@trace
def run_agent():
    handler = AgentDbgLangChainCallbackHandler()
    config = {"callbacks": [handler]}

    # Use config with any LangChain chain, LLM, or tool:
    result = my_chain.invoke(input_data, config=config)
    return result
```

The handler captures:

- **LLM calls** (`on_llm_start` / `on_chat_model_start` -> `on_llm_end`): records model name, prompt, response, and token usage via `record_llm_call`.
- **Tool calls** (`on_tool_start` -> `on_tool_end` / `on_tool_error`): records tool name, args, result, and error status via `record_tool_call`.

See `examples/langchain/minimal.py` for a runnable example:

```bash
uv run --extra langchain python examples/langchain/minimal.py
agentdbg view
```

**Guardrails (e.g. `stop_on_loop`) with LangChain / LangGraph:**
All guardrails work with the callback handler. When a guardrail fires, the handler sets `raise_error = True` and re-raises the exception, which tells LangChain to propagate it instead of swallowing it. This stops the graph mid-execution. See [Guardrails](guardrails.md) for details.

**Notes:**

- The handler requires an active AgentDbg run - wrap your entrypoint with `@trace` or set `AGENTDBG_IMPLICIT_RUN=1`.
- Tool errors are recorded as `TOOL_CALL` events with `status="error"` and include the error message.
- LLM errors are recorded as `LLM_CALL` events with `status="error"` (not as separate `ERROR` events).

---

### OpenAI Agents SDK tracing adapter

**Status: available.** An optional adapter lives at `agentdbg.integrations.openai_agents`. Importing it registers an OpenAI Agents tracing processor that forwards SDK generation, function, and handoff spans into the active AgentDbg run.

**Requirements:** `openai-agents` must be installed. Install the optional OpenAI dependency group (the `openai` group contains `openai-agents`):

```bash
pip install -e ".[openai]"
```

If `openai-agents` is not installed, importing the integration raises a clear `ImportError` with install instructions. The integration is optional; the core package does not depend on it.

**Usage:**

```python
from agentdbg import trace
from agentdbg.integrations import openai_agents  # registers hooks


@trace
def run_agent():
    # ... OpenAI Agents SDK code ...
    pass
```

The adapter captures:

- **LLM calls** (`GenerationSpanData`): records model, prompt, response, and usage via `record_llm_call`.
- **Tool calls** (`FunctionSpanData`): records tool name, args, result, and error status via `record_tool_call`.
- **Handoffs** (`HandoffSpanData`): records a `TOOL_CALL` named `handoff`, with framework-specific details stored in `meta`.

See `examples/openai_agents/minimal.py` for a runnable fake-data example:

```bash
uv run --extra openai python examples/openai_agents/minimal.py
agentdbg view
```

**Guardrails with OpenAI Agents SDK:**
The SDK wraps all tracing processor calls in `try/except` and logs errors, so guardrail exceptions cannot propagate to stop the run. When a guardrail fires, the exception is stored on `PROCESSOR.abort_exception`. Call `PROCESSOR.raise_if_aborted()` after `Runner.run()` to re-raise it:

```python
from agentdbg import trace, AgentDbgLoopAbort
from agentdbg.integrations.openai_agents import PROCESSOR

@trace(stop_on_loop=True)
async def run_agent():
    result = await Runner.run(agent, input)
    PROCESSOR.raise_if_aborted()
    return result
```

**Notes:**

- The adapter records events only while an explicit AgentDbg run is active; wrap your entrypoint with `@trace` or `traced_run(...)`.
- Framework-specific span details stay in `meta.openai_agents.*`, not the event payload.
- The example uses low-level SDK tracing spans with deterministic fake data, so it needs no API key and makes no model calls.

---

## Planned

Planned framework adapters (not yet implemented):

1. **Agno** - optional adapter for Agno-based agents.
2. Others as needed (e.g. AutoGen, CrewAI, custom loops).

For guidance on adding new integrations (optional deps, mapping callbacks to `record_*`, tests), see [CONTRIBUTING.md](../CONTRIBUTING.md#adding-integrations--adapters) in the repo root.

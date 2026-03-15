# Guardrails

Guardrails are AgentDbg's opt-in way to stop a run before it burns more time, tokens, or tool calls than you intended.

They are designed for local debugging loops, not policy enforcement:

- Use existing v0.1 event types only
- Record normal trace evidence before aborting
- Raise a dedicated exception so your code knows the run was stopped on purpose
- Keep default behavior unchanged unless you enable a guardrail

---

## What guardrails do

When a configured threshold is crossed, AgentDbg:

1. Records the relevant warning or event using the existing trace format
2. Raises `AgentDbgLoopAbort` or `AgentDbgGuardrailExceeded`
3. Records `ERROR`
4. Finalizes the run with `RUN_END` and `status="error"`

The resulting `ERROR` payload includes:

- `guardrail`
- `threshold`
- `actual`
- `error_type`
- `message`

That means the UI and raw trace both show not just that the run failed, but why it was intentionally stopped.

---

## Available guardrails

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `stop_on_loop` | `bool` | `False` | Abort when loop detection emits `LOOP_WARNING` |
| `stop_on_loop_min_repetitions` | `int` | `3` | Minimum repeated pattern count required to abort on loop |
| `max_llm_calls` | `int \| None` | `None` | Abort after more than N LLM calls |
| `max_tool_calls` | `int \| None` | `None` | Abort after more than N tool calls |
| `max_events` | `int \| None` | `None` | Abort after more than N total events |
| `max_duration_s` | `float \| None` | `None` | Abort when elapsed run time reaches the limit |

Notes:

- Count-based guardrails trigger at **N+1**, not at N.
- `max_duration_s` triggers when elapsed time is greater than or equal to the configured value.
- `stop_on_loop` does not create a new event type. It relies on the existing `LOOP_WARNING` event, then aborts.

---

## LangChain / LangGraph

Guardrails work with LangChain/LangGraph via `AgentDbgLangChainCallbackHandler`. When a guardrail fires inside a callback, the handler sets `raise_error = True` and re-raises, which tells LangChain to propagate the exception instead of swallowing it. This stops the graph mid-execution -- the same behavior as a direct `record_*` call.

```python
from agentdbg import AgentDbgLoopAbort, trace
from agentdbg.integrations import AgentDbgLangChainCallbackHandler


@trace(stop_on_loop=True, stop_on_loop_min_repetitions=3)
def run_agent():
    handler = AgentDbgLangChainCallbackHandler()
    return graph.invoke(state, config={"callbacks": [handler]})


try:
    run_agent()
except AgentDbgLoopAbort as exc:
    print(f"Stopped the loop: {exc}")
```

The handler also stores the exception on `handler.abort_exception` as a defensive fallback, with a `handler.raise_if_aborted()` convenience method.

## OpenAI Agents SDK

The OpenAI Agents SDK wraps all tracing processor calls in `try/except` and unconditionally logs errors -- there is no `raise_error` equivalent to force propagation. Guardrails still **detect** loops and record LOOP_WARNING, but the exception cannot stop the SDK's execution.

To react after the run finishes, check `PROCESSOR.raise_if_aborted()`:

```python
from agentdbg import trace, AgentDbgLoopAbort
from agentdbg.integrations.openai_agents import PROCESSOR


@trace(stop_on_loop=True)
async def run_agent():
    result = await Runner.run(agent, input)
    PROCESSOR.raise_if_aborted()
    return result


try:
    asyncio.run(run_agent())
except AgentDbgLoopAbort as exc:
    print(f"Loop detected: {exc}")
```

---

## Quick examples

### Stop a looping agent immediately

```python
from agentdbg import AgentDbgLoopAbort, record_llm_call, record_tool_call, trace


@trace(stop_on_loop=True)
def run_agent():
    for _ in range(10):
        record_tool_call("search_db", args={"q": "pricing"}, result={"hits": 3})
        record_llm_call(model="gpt-4.1", prompt="Summarize", response="Retrying...")


try:
    run_agent()
except AgentDbgLoopAbort as exc:
    print(f"Stopped because of a loop: {exc}")
```

### Cap LLM and tool usage during development

```python
from agentdbg import AgentDbgGuardrailExceeded, record_llm_call, record_tool_call, traced_run


try:
    with traced_run(
        name="react_debug",
        max_llm_calls=8,
        max_tool_calls=12,
        max_events=40,
        max_duration_s=30,
    ):
        # ... your agent loop ...
        record_llm_call(model="gpt-4.1", prompt="...", response="...")
        record_tool_call(name="search", args={"q": "docs"}, result={"hits": 2})
except AgentDbgGuardrailExceeded as exc:
    print(exc.guardrail, exc.threshold, exc.actual)
```

---

## Configuration surfaces

Guardrails can be configured in four places:

1. `@trace(...)`
2. `traced_run(...)`
3. Project or user YAML config
4. Environment variables

### Precedence

Highest wins:

1. Function arguments passed to `@trace(...)` or `traced_run(...)`
2. Environment variables
3. Project YAML: `.agentdbg/config.yaml`
4. User YAML: `~/.agentdbg/config.yaml`
5. Defaults

### Decorator and context manager

```python
from agentdbg import trace, traced_run


@trace(stop_on_loop=True, max_llm_calls=50)
def guarded_fn():
    ...


with traced_run(stop_on_loop=True, max_llm_calls=50):
    ...
```

### Environment variables

```bash
export AGENTDBG_STOP_ON_LOOP=1
export AGENTDBG_STOP_ON_LOOP_MIN_REPETITIONS=3
export AGENTDBG_MAX_LLM_CALLS=50
export AGENTDBG_MAX_TOOL_CALLS=50
export AGENTDBG_MAX_EVENTS=200
export AGENTDBG_MAX_DURATION_S=60
```

### YAML config

```yaml
# .agentdbg/config.yaml or ~/.agentdbg/config.yaml
guardrails:
  stop_on_loop: true
  stop_on_loop_min_repetitions: 3
  max_llm_calls: 50
  max_tool_calls: 50
  max_events: 200
  max_duration_s: 60
```

---

## How guardrails appear in traces

### Loop guardrail

If loop detection fires and `stop_on_loop=True`:

- AgentDbg first writes `LOOP_WARNING`
- Then raises `AgentDbgLoopAbort`
- Then writes `ERROR`
- Then writes `RUN_END(status="error")`

### Count and duration guardrails

For `max_llm_calls`, `max_tool_calls`, `max_events`, and `max_duration_s`:

- AgentDbg writes the event that crossed the limit
- Then raises `AgentDbgGuardrailExceeded`
- Then writes `ERROR`
- Then writes `RUN_END(status="error")`

This gives you full evidence of the step that actually tripped the limit.

---

## Choosing sensible defaults

Some practical starting points for local development:

- `stop_on_loop=True` for ReAct-style or planner/executor loops
- `max_llm_calls=10` to `30` for prompt iteration
- `max_tool_calls=10` to `25` for tool-heavy debugging
- `max_events=50` to `200` when you want a hard ceiling on trace size
- `max_duration_s=15` to `60` for runs that should finish quickly

Use tighter limits in tests and demos, and looser limits when you are intentionally exploring bigger workflows.

---

## Related docs

- [SDK](sdk.md)
- [Configuration reference](reference/config.md)
- [Trace format](reference/trace-format.md)
- [Architecture](architecture.md)

"""
debug_utils.py

Shared debug helpers for the agent-ladder scripts.

Every script in this project talks to the model the same way: it sends a
request (model + messages, sometimes tools) and gets a response back. When
you're learning, it's invaluable to SEE those raw payloads. Rather than
copy-paste the same debug code into every script, the logic lives here once.

Enable it per run with an environment variable:

    DEBUG=1 uv run python 01_chat_loop.py

Accepted truthy values: 1, true, yes (case-insensitive).

In debug mode you'll see the full round trip as labeled blocks:

    REQUEST            -> what we send to the model
    RESPONSE           -> what the model sends back (incl. tool_calls)
    TOOL EXECUTION     -> the function YOUR code ran between model calls
    REQUEST / RESPONSE -> the follow-up call that uses the tool result
"""

import json
import os

# Single source of truth for whether debug output is on.
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


def _to_serializable(obj):
    """json.dumps fallback that expands rich objects into plain JSON.

    ollama returns pydantic models (ChatResponse, Message, ToolCall, ...).
    Without this, json.dumps can't serialize them and falls back to a flat
    repr() string -- which hides the nested structure (e.g. the tool_calls
    array buried inside the assistant message). model_dump(mode="json")
    recursively turns them into proper dicts/lists with JSON-safe values.
    """
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    try:
        return dict(obj)
    except Exception:
        return str(obj)


def _print_block(label, payload):
    print(f"\n----- {label} -----")
    print(json.dumps(payload, indent=2, default=_to_serializable, ensure_ascii=False))
    print("-" * (12 + len(label)) + "\n")


def dump(label, payload):
    """Pretty-print a labeled payload as nested JSON (no-op unless DEBUG)."""
    if not DEBUG:
        return
    _print_block(label, payload)


def dump_request(model, messages, tools=None):
    """Dump an outgoing chat request the way it's sent to the model."""
    payload = {"model": model, "messages": messages}
    if tools is not None:
        payload["tools"] = tools
    dump("REQUEST", payload)


def dump_response(response):
    """Dump the raw response (message, tool_calls, token counts, timings)."""
    # Pass the object straight through; _to_serializable expands it (and its
    # nested message / tool_calls) into proper JSON instead of a repr string.
    dump("RESPONSE", response)


def tool_call(name, arguments, result):
    """Show a LOCAL tool execution -- the step the model can't do itself.

    The model only ASKED for this tool (see tool_calls in the RESPONSE above);
    your script is what actually runs it. The result shown here is exactly what
    gets fed back into the next REQUEST as a 'tool' message.

    Always prints (it's a real action, not just diagnostics): a full labeled
    block in debug mode, or a one-line summary otherwise.
    """
    if DEBUG:
        _print_block(
            "TOOL EXECUTION (local)",
            {"name": name, "arguments": arguments, "result": result},
        )
    else:
        print(f"  [tool call] {name}({arguments})")


def banner():
    """Print a one-line notice when debug mode is active."""
    if DEBUG:
        print("DEBUG mode ON: printing full request/response payloads.")

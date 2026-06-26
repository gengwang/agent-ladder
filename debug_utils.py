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

Optional `.env` in the project root is loaded on import (via python-dotenv).
Inline env vars still win — `MODEL=llama3.1:8b uv run ...` overrides `.env`.

In debug mode you'll see the full round trip as labeled blocks:

    REQUEST            -> what we send to the model
    RESPONSE           -> what the model sends back (incl. tool_calls)
    TOOL CALL          -> a tool the model asked for, logged BEFORE we run it
    TOOL RESULT        -> what that tool returned, logged after it finishes
    REQUEST / RESPONSE -> the follow-up call that uses the tool result

Tool logging is depth-aware: a tool whose body runs ANOTHER agent (rung 06)
prints its nested tool calls indented underneath it, in true execution order.
"""

import json
import os
from contextlib import contextmanager

from dotenv import load_dotenv

# Load optional .env from the project root. Does not override vars already set
# in the shell (so MODEL=... uv run ... still works).
load_dotenv()

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


# How deeply nested the current tool call is. 0 = a tool called by the top-level
# agent; 1 = a tool called by a sub-agent (rung 06), and so on. Used only to
# indent/label output so nested traces read in natural order.
_tool_depth = 0


def _tool_indent():
    # Base indent of two spaces (depth 0) matches every rung's original output;
    # each nesting level adds two more.
    return "  " * (_tool_depth + 1)


def _depth_suffix():
    return f" [depth {_tool_depth}]" if _tool_depth else ""


class _ToolCall:
    """Handle yielded by `tool_call`; set `.result` so it can be logged after."""

    __slots__ = ("name", "result")

    def __init__(self, name):
        self.name = name
        self.result = None


@contextmanager
def tool_call(name, arguments):
    """Wrap a LOCAL tool execution -- the step the model can't do itself.

    The model only ASKED for this tool (see tool_calls in the RESPONSE above);
    your script is what actually runs it. Used as a context manager so the call
    is logged BEFORE the tool runs and the result AFTER -- which means a tool
    whose body runs another agent (rung 06) shows its nested calls in true
    execution order, indented underneath it:

        with debug_utils.tool_call(name, args) as call:
            call.result = run_the_tool(args)

    The call line always prints (compact when DEBUG is off, a labeled block when
    on); the result prints as its own block only in debug mode. Depth is always
    restored, even if the tool raises.
    """
    global _tool_depth
    if DEBUG:
        _print_block(
            f"TOOL CALL (local){_depth_suffix()}",
            {"name": name, "arguments": arguments},
        )
    else:
        print(f"{_tool_indent()}[tool call] {name}({arguments})")

    record = _ToolCall(name)
    _tool_depth += 1
    try:
        yield record
    finally:
        _tool_depth -= 1
        if DEBUG:
            _print_block(
                f"TOOL RESULT (local){_depth_suffix()}",
                {"name": name, "result": record.result},
            )


def banner():
    """Print a one-line notice when debug mode is active."""
    if DEBUG:
        print("DEBUG mode ON: printing full request/response payloads.")

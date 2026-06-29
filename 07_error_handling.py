"""
07_error_handling.py

The same agent loop as 04 — but it survives things going wrong.

CONCEPT TO INTERNALIZE:
Every rung so far walked the happy path: the model call succeeds, the model
emits well-formed tool calls, and each tool runs cleanly. Real loops don't get
that luxury. Resilience is the difference between a demo and an agent, and it
comes down to sorting failures into two kinds and handling each correctly:

  1. TOOL failures the MODEL can see and fix. Malformed JSON arguments, a
     missing required field, a wrong tool name, or a tool that raises. The model
     CAN react to these — retry with corrected arguments, pick another tool, or
     tell the user. So we catch them and feed the error back AS THE TOOL RESULT,
     never letting them escape the loop.

  2. INFRASTRUCTURE failures the model CANNOT see. The model call itself fails:
     Ollama isn't running, a timeout, a dropped connection. The model can't
     observe or fix these, so retrying is OUR job — a few attempts with backoff,
     then degrade gracefully (abort just this turn, keep the REPL alive).

Two invariants make the loop trustworthy:

  - EVERY tool_call gets a matching tool result, even on failure. The protocol
    pairs calls with results (§4); a missing result leaves the conversation
    malformed and the next model call confused. An error string is still a
    result.
  - A failed turn must not corrupt the conversation. If a turn blows up
    mid-flight it can leave a dangling tool_call with no results. We checkpoint
    `messages` before the turn and roll back on failure, so the next turn always
    starts from a well-formed history.

Nothing about the loop's shape changes from 04. We're hardening the fragile
points — the model call, argument parsing, tool execution, and the prompt
itself — so one bad input can't take the whole agent down.
"""

import datetime
import json
import os
import time

import ollama

import cli_utils
import debug_utils

MODEL = os.environ.get("MODEL", "qwen3:8b")
MAX_STEPS = 10

# Model-call resilience knobs. A single ollama.chat may fail transiently
# (server loading, timeout, connection blip); retry a few times with growing
# backoff before giving up on the turn.
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds; doubles each attempt: 0.5, 1.0, 2.0, ...


class ModelUnavailable(Exception):
    """Raised when the model can't be reached after all retries are exhausted.

    Distinct from a tool error: this is an infrastructure failure the model
    can't see or fix, so it ends the current turn instead of being fed back.
    """


# ---- Tools: unchanged from 04. Resilience is the only new idea this rung. ----
def read_file(path: str) -> str:
    """Reads and returns the contents of a text file."""
    try:
        with open(os.path.expanduser(path), "r") as f:
            return f.read()
    except Exception as e:
        return f"ERROR reading file: {e}"


def list_dir(path: str) -> str:
    """Lists the entries in a directory, one per line."""
    try:
        entries = sorted(os.listdir(os.path.expanduser(path)))
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"ERROR listing directory: {e}"


def get_current_time() -> str:
    """Returns the current local date and time as a string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a single text file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read, e.g. ~/notes.txt",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List the files and folders inside a directory. Use this when "
                "the user asks what is IN a folder, not to read a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list, e.g. ~/Projects",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "get_current_time": get_current_time,
}


def call_model(messages, tools):
    """Call the model, retrying transient failures with exponential backoff.

    A model call can fail for reasons the MODEL can't see or fix (Ollama down,
    timeout, dropped connection), so recovering is the harness's job: retry a
    few times, then raise ModelUnavailable so the caller can end this turn
    without crashing the whole session.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            debug_utils.dump_request(MODEL, messages, tools=tools)
            response = ollama.chat(model=MODEL, messages=messages, tools=tools)
            debug_utils.dump_response(response)
            return response
        except Exception as e:
            # Last attempt failed -> hand control back to the turn handler.
            if attempt == MAX_RETRIES:
                raise ModelUnavailable(f"{type(e).__name__}: {e}") from e
            wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            print(
                f"  [model call failed ({type(e).__name__}: {e}); "
                f"attempt {attempt}/{MAX_RETRIES}, retrying in {wait:.1f}s]"
            )
            time.sleep(wait)


def _safe_invoke(fn_name, raw_args):
    """Run ONE tool defensively, returning its output or an ERROR string.

    Each failure mode below is something a model genuinely produces: a name we
    don't have, arguments that aren't valid JSON, arguments of the wrong shape,
    a missing/extra parameter, or a tool that raises internally. None of them
    should crash the loop — they become text the model can read and recover from.
    """
    fn = AVAILABLE_FUNCTIONS.get(fn_name)
    if fn is None:
        known = ", ".join(AVAILABLE_FUNCTIONS) or "(none)"
        return f"ERROR: unknown tool '{fn_name}'. Available tools: {known}."

    # Arguments arrive as a dict already, or as a JSON string a confused model
    # can mangle. Parse defensively before we ever unpack them.
    args = raw_args
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            return f"ERROR: arguments for '{fn_name}' were not valid JSON: {e}."
    if not isinstance(args, dict):
        return (
            f"ERROR: arguments for '{fn_name}' must be a JSON object of named "
            f"parameters, got {type(args).__name__}."
        )

    try:
        return fn(**args)
    except TypeError as e:
        # Wrong, missing, or extra parameter names: fn(**args) rejects them
        # before the tool body runs. Report it so the model can fix the call.
        return f"ERROR: bad arguments for '{fn_name}': {e}."
    except Exception as e:
        # The tool itself raised. Surface it as a result; never crash the loop.
        return f"ERROR: tool '{fn_name}' failed: {type(e).__name__}: {e}."


def run_tool_calls(tool_calls):
    """Execute every requested tool, yielding a result for EACH — even failures.

    The protocol pairs every tool_call with a tool result. _safe_invoke
    guarantees we always have a string to return, so the pairing holds no matter
    what the model asked for or how the tool behaved.
    """
    for call in tool_calls:
        fn_name = call["function"]["name"]
        raw_args = call["function"]["arguments"]
        with debug_utils.tool_call(fn_name, raw_args) as record:
            record.result = _safe_invoke(fn_name, raw_args)
        yield {"role": "tool", "name": fn_name, "content": record.result}


def run_turn(messages):
    """Drive the agent loop for one user turn and return its final text.

    May raise ModelUnavailable (model unreachable) — the caller is responsible
    for rolling back the conversation if that happens.
    """
    message = None
    for _step in range(MAX_STEPS):
        response = call_model(messages, TOOLS)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            break

        messages.extend(run_tool_calls(tool_calls))
    else:
        print(f"  [stopped: hit MAX_STEPS={MAX_STEPS} without a final answer]")

    return message.get("content", "") if message else ""


def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with three tools: read_file, "
                "list_dir, and get_current_time. Use as many tool calls as you "
                "need, in sequence, before giving your final answer. If a tool "
                "returns a message starting with 'ERROR', read it, fix your call "
                "(correct the arguments or choose another tool), and try again; "
                "if you truly can't proceed, explain the problem to the user."
            ),
        }
    ]

    print(f"Resilient agent running with model={MODEL}. Type 'exit' to quit.")
    debug_utils.banner()
    print()

    while True:
        # Hardened prompt read: Ctrl-D (EOF) or Ctrl-C at the prompt should quit
        # cleanly, not dump a traceback.
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cli_utils.is_exit_command(user_input):
            break
        if not user_input:
            continue

        # Checkpoint BEFORE the turn. If the turn fails partway, we roll back to
        # here so a dangling tool_call (a call with no result) can't corrupt the
        # next request.
        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})

        try:
            assistant_text = run_turn(messages)
        except ModelUnavailable as e:
            del messages[checkpoint:]
            print(f"Assistant: [model unavailable after {MAX_RETRIES} attempts: {e}]")
            print("  [turn rolled back; try again in a moment]\n")
            continue
        except KeyboardInterrupt:
            # Let the user abort a long or looping turn without quitting the app.
            del messages[checkpoint:]
            print("\n  [interrupted this turn; rolled back to the prompt]\n")
            continue

        print(f"Assistant: {assistant_text}\n")


if __name__ == "__main__":
    main()

"""
05_persistent_memory.py

Same agent loop as 04 — but the conversation survives when you quit and restart.

CONCEPT TO INTERNALIZE:
Rungs 01–04 keep the entire conversation in a Python list (`messages`) that
lives only in RAM. Exit the script and that list is gone — the model has
"forgotten" everything, even though nothing about the model changed. The model
was never remembering; the *client* was holding history and throwing it away.

Persistent memory is therefore not a model feature. It is the harness choosing
to SAVE the client-side messages list somewhere durable (here, a JSON file on
disk) and RELOAD it on the next run. Every API call is still stateless: we
still resend the whole conversation each turn. We just don't start from scratch
when the process restarts.

One deliberate split: the SYSTEM PROMPT is owned by the code, not by the file.
We persist only the conversation (user/assistant/tool turns) and re-inject a
fresh system prompt on load. That keeps the prompt authoritative — edit it here
and the change takes effect next run, instead of being shadowed by a stale copy
that got saved to disk on the first run.

Try this experiment:
  1. Run the script, tell it your name, type exit.
  2. Run the script again, ask "what's my name?"
     -> it works, because the saved conversation was loaded and resent.
  3. Delete `.agent_memory.json` and run again
     -> amnesia returns. The file WAS the memory.

Nothing else changes from 04: same tools, same agent loop, same termination
rule. The only new lines are load-at-start and save-after-each-turn.
"""

import datetime
import json
import os

import ollama

import cli_utils
import debug_utils

MODEL = os.environ.get("MODEL", "qwen3:8b")
MAX_STEPS = 10

# Where the messages list is written between runs. Override to put it elsewhere:
#   MEMORY_FILE=~/my-session.json uv run python 05_persistent_memory.py
MEMORY_FILE = os.path.expanduser(
    os.environ.get("MEMORY_FILE", ".agent_memory.json")
)

DEFAULT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful assistant with three tools: read_file, list_dir, "
        "and get_current_time. Use as many tool calls as you need, in sequence, "
        "before giving your final answer. When you have enough information, "
        "answer the user directly."
    ),
}


def _to_plain(obj):
    """Turn ollama/pydantic objects into JSON-safe dicts for persistence."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    return obj


def load_messages():
    """Rebuild the messages list: a FRESH system prompt + saved conversation.

    The system prompt is re-injected from code on every run, NOT loaded from
    disk. Only the conversation (user/assistant/tool turns) is persisted. This
    keeps the system prompt authoritative: edit it here and the change takes
    effect next run, instead of being shadowed by a stale copy saved to disk.
    """
    history = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, encoding="utf-8") as f:
            history = json.load(f)

    # Drop any persisted system turns; the current code owns the system prompt.
    history = [m for m in history if m.get("role") != "system"]
    return [DEFAULT_SYSTEM_MESSAGE.copy(), *history]


def save_messages(messages):
    """Persist the conversation — everything EXCEPT the system prompt.

    The system message is owned by the code (re-injected on load), so it's
    excluded here. What survives across runs is the actual memory: the
    user/assistant/tool turns.
    """
    plain = [_to_plain(m) for m in messages if _role_of(m) != "system"]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(plain, f, indent=2, ensure_ascii=False)


def _role_of(message):
    """Read the role from a dict or an ollama Message object."""
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


# ---- Tools: unchanged from 04. Persistence is the only new idea this rung. ----
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


def run_tool_calls(tool_calls):
    """Execute every tool the model asked for and append each result."""
    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = call["function"]["arguments"]
        if isinstance(fn_args, str):
            fn_args = json.loads(fn_args)

        fn = AVAILABLE_FUNCTIONS.get(fn_name)
        result = fn(**fn_args) if fn else f"ERROR: unknown tool {fn_name}"

        debug_utils.tool_call(fn_name, fn_args, result)

        yield {"role": "tool", "name": fn_name, "content": result}


def main():
    messages = load_messages()
    prior_turns = len(messages) - 1  # exclude system prompt

    print(f"Persistent-memory agent running with model={MODEL}. Type 'exit' to quit.")
    if prior_turns > 0:
        print(
            f"Loaded {prior_turns} saved message(s) from {MEMORY_FILE} "
            f"(delete that file to reset memory)."
        )
    debug_utils.banner()
    print()

    while True:
        user_input = input("You: ").strip()
        if cli_utils.is_exit_command(user_input):
            break

        messages.append({"role": "user", "content": user_input})

        message = None
        for step in range(MAX_STEPS):
            debug_utils.dump_request(MODEL, messages, tools=TOOLS)
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            debug_utils.dump_response(response)
            message = response["message"]

            messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                break

            messages.extend(run_tool_calls(tool_calls))
        else:
            print(f"  [stopped: hit MAX_STEPS={MAX_STEPS} without a final answer]")

        assistant_text = message.get("content", "") if message else ""
        print(f"Assistant: {assistant_text}\n")

        # Persist after each complete user turn — tool rounds included.
        save_messages(messages)


if __name__ == "__main__":
    main()

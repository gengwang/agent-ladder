"""
04_agent_loop.py

Stop hardcoding ONE tool round. Loop until the model says it's done.

CONCEPT TO INTERNALIZE:
This is THE agent loop -- the thing that makes a script "agentic." Rungs 02
and 03 allowed exactly one tool round per user turn: call the model, run any
tools it asked for, call the model once more, print. That breaks the moment a
task needs tools to run in SEQUENCE, where the second call depends on the
first's result. For example:
  "What is in the most recently named file in ~/Projects/agent-ladder?"
needs list_dir FIRST, and only after seeing that result can the model know
which file to read_file. One fixed round can't express that.

The fix is to stop counting rounds and instead loop on a CONDITION:

  keep (call model -> run requested tools -> feed results back)
  until the model returns a message with NO tool_calls.

A message with no tool_calls IS the model's signal "I have everything I need;
here is my final answer." That single rule turns a one-shot exchange into an
open-ended agent that can chain as many tool steps as the task requires --
read a dir, pick a file, read it, summarize -- all on its own.

Two details this rung also settles:
  - We append the assistant message after EVERY model call (tool-calling or
    not), which unifies 03's two separate append sites into one.
  - A real loop needs a termination guard. "No tool_calls" is the normal exit;
    MAX_STEPS bounds the pathological case where a confused model keeps asking
    forever. (Making the loop genuinely resilient is rung 07's job; this is
    just the seatbelt.)
"""

import datetime
import json
import os

import ollama

import cli_utils
import debug_utils

MODEL = os.environ.get("MODEL", "qwen3:8b")

# The loop's safety bound. The NORMAL way out is the model returning no
# tool_calls; this only catches a model that never stops asking for tools.
MAX_STEPS = 10


# ---- Tools: unchanged from 03. The loop is the only new idea this rung. ----
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
    """Execute every tool the model asked for and append each result.

    Identical dispatch to 02/03 -- pulled into a helper only because the agent
    loop now calls it from inside a loop instead of once. The model can only
    ASK; this is the code that actually runs the function and reports back.
    """
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
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with three tools: read_file, "
                "list_dir, and get_current_time. Use as many tool calls as you "
                "need, in sequence, before giving your final answer. When you "
                "have enough information, answer the user directly."
            ),
        }
    ]

    print(f"Agent-loop agent running with model={MODEL}. Type 'exit' to quit.")
    debug_utils.banner()
    print()

    while True:
        user_input = input("You: ").strip()
        if cli_utils.is_exit_command(user_input):
            break

        messages.append({"role": "user", "content": user_input})

        # ---- THE AGENT LOOP ----
        # Keep letting the model work until it returns a message with no
        # tool_calls. Each pass is one model call; a pass that asks for tools
        # runs them, feeds the results back, and loops so the model can use
        # them to decide its NEXT step (or finish).
        message = None
        for step in range(MAX_STEPS):
            debug_utils.dump_request(MODEL, messages, tools=TOOLS)
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            debug_utils.dump_response(response)
            message = response["message"]

            # Always record the assistant turn (whether it's a tool request or
            # the final answer) -- one append site replaces 03's two.
            messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                # No tool_calls == "I'm done." This is the normal exit.
                break

            messages.extend(run_tool_calls(tool_calls))
        else:
            # Ran MAX_STEPS times and the model was STILL asking for tools.
            print(f"  [stopped: hit MAX_STEPS={MAX_STEPS} without a final answer]")

        assistant_text = message.get("content", "") if message else ""
        print(f"Assistant: {assistant_text}\n")


if __name__ == "__main__":
    main()

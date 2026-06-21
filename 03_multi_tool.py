"""
03_multi_tool.py

Give the model a MENU of tools instead of just one, and watch it ROUTE.

CONCEPT TO INTERNALIZE:
"Tool selection" (a.k.a. routing) is the model's job, not yours. Once you
hand it more than one tool, every turn the model has to decide:
  - Do I need a tool at all? (or just answer from what I know)
  - If so, WHICH one? (read_file vs list_dir vs get_current_time)
  - With WHAT arguments?

You never write an if/elif tree that says "if the user mentions a file, call
read_file." The model picks. Your code's only job is to (a) describe each tool
clearly enough that the model can choose well, and (b) dispatch whatever it
asks for to the right Python function.

The big reveal of this rung: the DISPATCH LOOP from 02 did not change at all.
It already looped over `tool_calls` and looked each name up in a dict. Going
from one tool to many is almost entirely "add more entries to the menu" --
the routing intelligence lives in the model, and the plumbing was generic the
whole time. Description quality is now your main lever: vague descriptions =
bad routing.
"""

import datetime
import json
import os

import ollama

import cli_utils
import debug_utils

MODEL = os.environ.get("MODEL", "qwen3:8b")


# ---- Step A: the actual Python functions the tools will run ----
# Three deliberately DIFFERENT tools so the routing decision is visible:
# two touch the filesystem (and differ only in intent), one is unrelated.
def read_file(path: str) -> str:
    """Reads and returns the contents of a text file."""
    try:
        with open(os.path.expanduser(path), "r") as f:
            return f.read()
    except Exception as e:
        # Tool errors come back as text the MODEL can read and react to,
        # not as a Python exception that crashes the script.
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


# ---- Step B: describe each function to the model as a JSON schema ----
# This list IS the menu. The model reads these descriptions to route. Note
# read_file and list_dir both take a `path` and both touch the filesystem --
# the DESCRIPTIONS are what let the model tell "read a file" from "list a
# folder" apart. get_current_time takes no arguments at all.
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

# Map tool name -> actual Python function. This dict is what turns
# "the model asked for 'list_dir'" into "run this real function". Adding a
# tool is: write the function, add a schema entry above, add a line here.
AVAILABLE_FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "get_current_time": get_current_time,
}


def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with three tools: read_file, "
                "list_dir, and get_current_time. Pick the right tool for the "
                "user's request, or just answer directly if no tool is needed."
            ),
        }
    ]

    print(f"Multi-tool agent running with model={MODEL}. Type 'exit' to quit.")
    debug_utils.banner()
    print()

    while True:
        user_input = input("You: ").strip()
        if cli_utils.is_exit_command(user_input):
            break

        messages.append({"role": "user", "content": user_input})

        # First call: give the model the chance to route to a tool.
        debug_utils.dump_request(MODEL, messages, tools=TOOLS)
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        debug_utils.dump_response(response)
        message = response["message"]

        tool_calls = message.get("tool_calls")

        if tool_calls:
            # The model routed to one or more tools. Append its tool-call
            # message first so it can later see what IT asked for.
            messages.append(message)

            # This loop is UNCHANGED from 02 -- it was generic all along.
            # The model may even pick several different tools in one turn;
            # the same dict lookup dispatches each to the right function.
            for call in tool_calls:
                fn_name = call["function"]["name"]
                fn_args = call["function"]["arguments"]
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)

                fn = AVAILABLE_FUNCTIONS.get(fn_name)
                result = fn(**fn_args) if fn else f"ERROR: unknown tool {fn_name}"

                debug_utils.tool_call(fn_name, fn_args, result)

                # Feed the result back as a 'tool' message. We include `name`
                # so that, with several tools in play, the model can correlate
                # each result to the call it made.
                messages.append(
                    {
                        "role": "tool",
                        "name": fn_name,
                        "content": result,
                    }
                )

            # Second call: the model now has the tool result(s) and can
            # form its actual answer to the user.
            debug_utils.dump_request(MODEL, messages, tools=TOOLS)
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            debug_utils.dump_response(response)
            message = response["message"]

        assistant_text = message.get("content", "")
        print(f"Assistant: {assistant_text}\n")
        messages.append(message)


if __name__ == "__main__":
    main()

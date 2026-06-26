"""
06_two_agents.py

One agent calls another. A sub-agent is just a TOOL to the agent above it.

CONCEPT TO INTERNALIZE:
Orchestration sounds like a new mechanism. It isn't. A sub-agent is exposed to
the parent through the exact same tool-calling protocol from 02: the parent
"asks" for a tool, your code runs it, the result comes back. The only twist is
what that tool DOES — instead of reading a file, it runs a whole second agent
loop (its own model calls, its own tools) and returns that agent's final text.

So the new idea is a boundary, not a new control flow:

  - Two SEPARATE contexts. The orchestrator has its own `messages`; the
    researcher builds a fresh `messages` on every call. Neither sees the
    other's history. The researcher cannot read the orchestrator's
    conversation, and the orchestrator never sees the researcher's internal
    tool calls or reasoning — only the final string it returns.
  - That isolation is the whole point. Each agent gets a clean, focused context
    instead of one giant transcript. The orchestrator delegates a
    self-contained task; the worker solves it and reports back a summary.

Because the contexts are separate, the orchestrator must pass everything the
researcher needs INSIDE the question — the worker has no access to what the user
said earlier. Delegation is communication across a boundary, not shared memory.

Here the orchestrator has NO file tools at all; its only tool is `ask_researcher`.
To answer anything about the filesystem it is forced to delegate, which makes the
two-agent handoff impossible to miss.

This is the seed of multi-agent systems: planners calling workers, a "manager"
fanning out to specialists. They're all this same pattern — an agent whose tools
happen to be other agents.

The orchestrator also carries its own long-term memory (recall_memory / remember)
so it can look up facts like the user's name instead of asking the researcher to
scan the filesystem for them. That contrasts two kinds of memory: rung 05 saved
the WHOLE conversation automatically (implicit); here the model chooses what to
keep and consults it on demand (explicit, tool-based). Memory is the persistent
orchestrator's own faculty; real-world lookups still get delegated.
"""

import datetime
import json
import os

import ollama

import cli_utils
import debug_utils

MODEL = os.environ.get("MODEL", "qwen3:8b")
MAX_STEPS = 10

# Long-term memory for the orchestrator: a flat list of facts it chose to keep,
# persisted as JSON so it survives across runs. Override the path with NOTES_FILE.
NOTES_FILE = os.path.expanduser(os.environ.get("NOTES_FILE", ".agent_memory.json"))


# ---- The generic agent loop, factored out so BOTH agents can reuse it. ----
# This is exactly rung 04's loop, with the tools + dispatch map passed in. That
# it serves the orchestrator and the researcher unchanged is the lesson: an
# "agent" is just (messages + tools + this loop). Stacking them is orchestration.
def run_tool_calls(tool_calls, available_functions):
    """Execute every tool the model asked for and yield each result message."""
    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = call["function"]["arguments"]
        if isinstance(fn_args, str):
            fn_args = json.loads(fn_args)

        fn = available_functions.get(fn_name)
        # Log the call BEFORE running it. Because ask_researcher's body runs a
        # whole sub-agent, its nested tool calls print indented underneath it,
        # in true outer -> inner order.
        with debug_utils.tool_call(fn_name, fn_args) as call:
            call.result = fn(**fn_args) if fn else f"ERROR: unknown tool {fn_name}"

        yield {"role": "tool", "name": fn_name, "content": call.result}


def run_agent_loop(messages, tools, available_functions):
    """Drive one agent to completion and return its final text answer.

    Mutates `messages` in place (each model + tool turn is appended), so the
    caller keeps the full transcript. Returns only the final assistant text --
    which is all a PARENT agent gets to see when this loop is a sub-agent.
    """
    message = None
    for _step in range(MAX_STEPS):
        debug_utils.dump_request(MODEL, messages, tools=tools)
        response = ollama.chat(model=MODEL, messages=messages, tools=tools)
        debug_utils.dump_response(response)
        message = response["message"]

        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            break

        messages.extend(run_tool_calls(tool_calls, available_functions))
    else:
        print(f"  [stopped: hit MAX_STEPS={MAX_STEPS} without a final answer]")

    return message.get("content", "") if message else ""


# ---- The RESEARCHER sub-agent: the worker tools from 04 live down here. ----
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


RESEARCHER_SYSTEM = (
    "You are a research assistant with three tools: read_file, list_dir, and "
    "get_current_time. Answer the single question you are given using those "
    "tools, then reply with a concise, self-contained summary. You cannot ask "
    "follow-up questions, so work only from what the question states."
)

RESEARCHER_TOOLS = [
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

RESEARCHER_FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "get_current_time": get_current_time,
}


def ask_researcher(question: str) -> str:
    """SUB-AGENT entry point -- exposed to the orchestrator as one tool.

    This is the whole orchestration trick: from the orchestrator's side this is
    just a function that takes a string and returns a string, identical in shape
    to read_file. Inside, it runs a COMPLETE second agent loop with its OWN
    fresh context (`messages`) and its OWN tools. The orchestrator never sees
    any of that -- only the returned summary.
    """
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "user", "content": question},
    ]
    return run_agent_loop(messages, RESEARCHER_TOOLS, RESEARCHER_FUNCTIONS)


# ---- The orchestrator's OWN long-term memory, exposed as two tools. ----
# This is a different kind of memory from rung 05. There, the WHOLE conversation
# was saved and resent automatically (implicit memory). Here the model decides
# what is worth keeping and looks it up on demand -- EXPLICIT, tool-based memory.
# It belongs to the orchestrator (the persistent agent), not the throwaway
# researcher, and it's separate from delegation: facts about the user live in
# memory; facts about the world get delegated to the researcher.
def _load_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    try:
        with open(NOTES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def recall_memory() -> str:
    """Return every fact saved to long-term memory."""
    notes = _load_notes()
    if not notes:
        return "(memory is empty -- nothing has been remembered yet)"
    return "\n".join(f"- {note}" for note in notes)


def remember(note: str) -> str:
    """Append one durable fact to long-term memory (survives across runs)."""
    notes = _load_notes()
    notes.append(note)
    try:
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
    except OSError as e:
        return f"ERROR saving memory: {e}"
    return f"Remembered: {note}"


# ---- The ORCHESTRATOR: its own memory + the power to delegate research. ----
ORCHESTRATOR_SYSTEM = (
    "You are a coordinator with a long-term memory and a research sub-agent. "
    "You have NO direct access to files, directories, or the system clock.\n"
    "- Memory: call recall_memory to look up facts you saved earlier (e.g. the "
    "user's name or preferences), and remember to save a new durable fact. When "
    "the user tells you something personal worth keeping, call remember. When "
    "asked about something personal, call recall_memory FIRST instead of "
    "guessing or searching the filesystem.\n"
    "- Research: call ask_researcher for anything about the real world -- files, "
    "directories, or the current time. Send ONE self-contained question; the "
    "researcher cannot see this conversation, so include all needed details "
    "(full paths, exactly what to find).\n"
    "Then synthesize what you found into an answer for the user."
)

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Look up facts you previously saved to long-term memory (such as "
                "the user's name or preferences). Takes no arguments; returns all "
                "saved facts. Use this before assuming you don't know something "
                "personal about the user."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a single durable fact to long-term memory so it survives "
                "across sessions. Use for stable personal facts the user shares, "
                "e.g. their name or preferences -- not for transient details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": (
                            "The fact to remember, as a short self-contained "
                            "sentence, e.g. \"The user's name is Geng.\""
                        ),
                    }
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_researcher",
            "description": (
                "Delegate a single, self-contained question to a research "
                "sub-agent that can read files, list directories, and check the "
                "current time. The sub-agent has no memory of this conversation, "
                "so include every detail it needs (e.g. full file paths)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "A complete, standalone question for the researcher, "
                            "e.g. 'List the files in ~/Projects/agent-ladder and "
                            "report how many there are.'"
                        ),
                    }
                },
                "required": ["question"],
            },
        },
    },
]

ORCHESTRATOR_FUNCTIONS = {
    "recall_memory": recall_memory,
    "remember": remember,
    "ask_researcher": ask_researcher,
}


def main():
    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM}]

    print(f"Two-agent orchestrator running with model={MODEL}. Type 'exit' to quit.")
    print("It delegates real-world lookups to a researcher and keeps its own memory.")
    print(f"(Long-term memory: {NOTES_FILE} -- delete it to forget everything.)")
    debug_utils.banner()
    print()

    while True:
        user_input = input("You: ").strip()
        if cli_utils.is_exit_command(user_input):
            break

        messages.append({"role": "user", "content": user_input})

        # One orchestrator turn. Any ask_researcher call inside this loop spins
        # up a full researcher loop (its own context) before control returns.
        assistant_text = run_agent_loop(
            messages, ORCHESTRATOR_TOOLS, ORCHESTRATOR_FUNCTIONS
        )
        print(f"Assistant: {assistant_text}\n")


if __name__ == "__main__":
    main()

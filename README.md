# agent-ladder

A learning ladder for understanding agent harnesses from first principles,
before reaching for a framework like Smolagents or LangGraph.

Each script is self-contained. Read the docstring at the top of each file
before running it -- that's where the concept being taught lives.

## Setup

### Option A: uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtual environment and Python
version for you.

```bash
cd ~/Projects/agent-ladder
uv venv
uv pip install -r requirements.txt
```

### Option B: stdlib venv + pip

```bash
cd ~/Projects/agent-ladder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make sure Ollama is running locally and you have a tool-calling-capable
model pulled, e.g.:

```bash
ollama pull qwen3:8b
```

(qwen3:8b supports tool calling. If you use a different model, confirm it
supports tools -- not all do.)

## Running

With uv (no need to activate the venv -- `uv run` uses it automatically):

```bash
uv run python 01_chat_loop.py
uv run python 02_single_tool.py
uv run python 03_multi_tool.py
uv run python 04_agent_loop.py
uv run python 05_persistent_memory.py
uv run python 06_two_agents.py
uv run python 07_error_handling.py
```

Or, with an activated venv:

```bash
python3 01_chat_loop.py
python3 02_single_tool.py
python3 03_multi_tool.py
python3 04_agent_loop.py
python3 05_persistent_memory.py
python3 06_two_agents.py
python3 07_error_handling.py
```

### Debug mode

Set `DEBUG=1` to print the full request (the entire conversation history sent
to the model every turn) and the raw response (message plus token counts and
timings). This is the clearest way to *see* the statelessness concept --
watch the `messages` array grow with each turn. See [`LEARNING.md`](./LEARNING.md)
for what those token counts, timings, and `tool_calls` fields actually mean.

```bash
DEBUG=1 uv run python 01_chat_loop.py
```

Accepted truthy values: `1`, `true`, `yes` (case-insensitive).

Tool calls are logged the moment the model asks for them, *before* they run, so
the trace reads in true execution order. When a tool's body runs another agent
(rung 06), its nested tool calls print indented underneath it:

```
  [tool call] ask_researcher({...})
    [tool call] get_current_time({})
```

### Quitting

Every script shares one exit check (`cli_utils.is_exit_command`), so the same
words work everywhere. At the `You:` prompt, type `exit`, `quit`, or `/bye`
(case-insensitive, surrounding whitespace ignored) to end the session.

### Model selection

Default model is `qwen3:8b`. Override per run or via a `.env` file (copy
`.env.example` to `.env`):

```bash
MODEL=qwen3.5:latest uv run python 01_chat_loop.py
```

Inline env vars win over `.env` — useful when you want a one-off model without
editing the file.

For 02, try asking something like:
  "what's in ~/Projects/agent-ladder/requirements.txt?"
and watch the `[tool call]` line print before the model's answer --
that's you seeing the tool-call protocol happen in real time.

For 03, the model now picks *which* tool to use. Try each in turn and watch
which tool the `[tool call]` line names:
  "what files are in ~/Projects/agent-ladder?"   (-> list_dir)
  "read ~/Projects/agent-ladder/requirements.txt" (-> read_file)
  "what time is it?"                               (-> get_current_time)
  "what's the capital of France?"                  (-> no tool, direct answer)

For 04, the model can chain tools across multiple steps in one turn. Ask
something that needs a SEQUENCE and watch several `[tool call]` lines print
before the answer:
  "list ~/Projects/agent-ladder, then read its README and summarize it."
That's list_dir -> (model picks a file) -> read_file -> final answer, all
driven by the loop, not a hardcoded second call.

For 05, memory survives across runs. Tell it your name, type exit, start the
script again, and ask "what's my name?" — the saved history is loaded and
resent. Delete `.agent_memory.json` to reset.

For 06, the orchestrator delegates real-world lookups to a researcher sub-agent
and keeps its own long-term memory. Ask something that needs the filesystem and
watch the researcher's own `[tool call]` lines fire inside the `ask_researcher`
call:
  "how many files are in ~/Projects/agent-ladder, and what's in its README?"
That's the orchestrator delegating -> the researcher running its own agent loop
(separate context) -> returning just a summary the orchestrator synthesizes.

To see the memory tools, tell it a fact and then ask for it back -- it saves with
`remember` and looks up with `recall_memory` instead of searching the system:
  "my name is Geng"        (-> remember)
  "what's my name?"        (-> recall_memory)
Memory persists to `.agent_notes.json`; delete that file to make it forget.

For 07, the loop survives things going wrong instead of crashing. Two ways to
see it:
  - Tool errors become results the model recovers from. Ask it to read a file
    that doesn't exist:
      "read ~/Projects/agent-ladder/nope.txt"
    and watch the `ERROR ...` come back as the tool result, then the model
    react to it rather than the script dying.
  - Infrastructure errors retry, then degrade. Stop Ollama (`ollama stop` or
    quit the app) and send any message: the model call retries with backoff and,
    after `MAX_RETRIES`, the turn rolls back with a readable message while the
    prompt stays alive. Ctrl-C mid-turn aborts just that turn; Ctrl-C/Ctrl-D at
    the prompt quits cleanly.

## The ladder

1. `01_chat_loop.py` -- plain chat, no tools. Statelessness.
2. `02_single_tool.py` -- one tool (read_file). The tool-call protocol.
3. `03_multi_tool.py` -- multiple tools. Tool selection/routing.
4. `04_agent_loop.py` -- repeated tool calls until done. The real agent loop.
5. `05_persistent_memory.py` -- state surviving across script runs.
6. `06_two_agents.py` -- one agent calling another. Orchestration.
7. `07_error_handling.py` -- making the loop resilient, not just functional.
8. `08_smolagents_version.py` -- same task, rebuilt in Smolagents. (next)
9. `09_langgraph_version.py` -- same task, rebuilt in LangGraph.

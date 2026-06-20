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
```

Or, with an activated venv:

```bash
python3 01_chat_loop.py
python3 02_single_tool.py
```

### Debug mode

Set `DEBUG=1` to print the full request (the entire conversation history sent
to the model every turn) and the raw response (message plus token counts and
timings). This is the clearest way to *see* the statelessness concept --
watch the `messages` array grow with each turn.

```bash
DEBUG=1 uv run python 01_chat_loop.py
```

Accepted truthy values: `1`, `true`, `yes` (case-insensitive).

For 02, try asking something like:
  "what's in ~/Projects/agent-ladder/requirements.txt?"
and watch the `[tool call]` line print before the model's answer --
that's you seeing the tool-call protocol happen in real time.

## The ladder

1. `01_chat_loop.py` -- plain chat, no tools. Statelessness.
2. `02_single_tool.py` -- one tool (read_file). The tool-call protocol.
3. `03_multi_tool.py` -- multiple tools. Tool selection/routing. (next)
4. `04_agent_loop.py` -- repeated tool calls until done. The real agent loop.
5. `05_persistent_memory.py` -- state surviving across script runs.
6. `06_two_agents.py` -- one agent calling another. Orchestration.
7. `07_error_handling.py` -- making the loop resilient, not just functional.
8. `08_smolagents_version.py` -- same task, rebuilt in Smolagents.
9. `09_langgraph_version.py` -- same task, rebuilt in LangGraph.

# AGENTS.md

Working agreement for AI agents (and humans) contributing to `agent-ladder`.
This file is auto-discovered — read it first.

This project is a **teaching ladder**: each numbered script climbs one rung,
introducing exactly one new agent-harness concept from first principles. The
code is the lesson, and [`LEARNING.md`](./LEARNING.md) is the running concept
companion.

---

## Golden rule

> **When you add or meaningfully change a script, update `LEARNING.md`
> (and `README.md` where user-facing) in the same change.** The repo's value is
> the pairing of runnable code with the concepts behind it. A script without its
> concept notes is an incomplete rung.

If you do only one thing from this document, do that.

---

## The ladder

Scripts are numbered `NN_name.py` and meant to be read/run in order. Each rung
teaches ONE new idea and reuses everything below it.

1. `01_chat_loop.py` — plain chat, no tools. Statelessness.
2. `02_single_tool.py` — one tool (read_file). The tool-call protocol.
3. `03_multi_tool.py` — multiple tools. Tool selection/routing.
4. `04_agent_loop.py` — repeated tool calls until done. The real agent loop.
5. `05_persistent_memory.py` — state surviving across script runs.
6. `06_two_agents.py` — one agent calling another. Orchestration.
7. `07_error_handling.py` — making the loop resilient, not just functional.
8. `08_smolagents_version.py` — same task, rebuilt in Smolagents.
9. `09_langgraph_version.py` — same task, rebuilt in LangGraph.

---

## Checklist: adding a new rung

- [ ] **Docstring first.** Open with a docstring stating the ONE concept taught,
      plus a short "CONCEPT TO INTERNALIZE" note, matching `01`/`02` style.
- [ ] **Reuse, don't duplicate.** Layer the new idea on top of lower rungs.
- [ ] **Wire in shared debug** via `debug_utils` (see Conventions).
- [ ] **Update `LEARNING.md`** — add a section + table-of-contents entry for the
      new concept, ideally with real numbers/output you saw while testing.
- [ ] **Update `README.md`** — keep "The ladder" list and Running/Debug sections
      accurate (drop any `(next)` marker, add run examples).
- [ ] **Keep deps honest** — add new dependencies to `requirements.txt` (pinned
      where reasonable) and mention them in `README.md`.
- [ ] **Lint clean** — no linter errors in files you touched.

## Checklist: changing an existing script

- [ ] If behavior changes or a new detail is illustrated, reflect it in
      `LEARNING.md` (and `README.md` if user-facing).
- [ ] Preserve "one concept per rung" — resist creeping into the next lesson.

---

## Conventions

### Shared debug helpers (`debug_utils.py`)

Every script that calls the model should surface the request/response so a
student can see the internals. Use the shared helpers — do NOT scatter
`if DEBUG:` blocks or local dump functions per script.

```python
import debug_utils

debug_utils.banner()                                    # at startup
debug_utils.dump_request(MODEL, messages, tools=TOOLS)  # before a model call
response = ollama.chat(...)
debug_utils.dump_response(response)                     # after a model call
debug_utils.tool_call(name, arguments, result)          # when YOUR code runs a tool
```

- Toggled by the `DEBUG` env var (`1`/`true`/`yes`), centralized as
  `debug_utils.DEBUG`. Keep it env-driven so code needs no edits to toggle.
- The dump serializer expands pydantic objects (ollama's
  `ChatResponse`/`Message`/`ToolCall`) into nested JSON. New rich objects you
  want printed should serialize cleanly (expose `model_dump()` or pass dicts).
- `dump_*` are no-ops unless `DEBUG` is on; `tool_call` always prints (compact
  line when off, full labeled block when on).

### Style

- Self-contained scripts: a reader should understand a rung from that one file
  plus `debug_utils`.
- Comments explain *why*/the concept, not line-by-line *what*.
- Default model is `qwen3:8b` (tool-calling capable). If a rung needs a
  different model, say so in its docstring.

### Running

Use `uv`, from the project root (so `debug_utils` imports resolve):

```bash
uv run python NN_name.py
DEBUG=1 uv run python NN_name.py
```

---

## Why this matters

The premise of the ladder is that frameworks (Smolagents, LangGraph) are
demystified once you've built the loop by hand. That only works if every rung
leaves behind a clear, accurate concept note. Treat letting `LEARNING.md` drift
out of date as a bug.

# LEARNING.md

Notes captured while building and exploring the `agent-ladder` scripts. This is
a study companion to the code — it records the *concepts* behind what the
scripts demonstrate, written as a running Q&A with myself.

## Table of contents

1. [Environment & tooling (uv)](#1-environment--tooling-uv)
2. [Debug mode](#2-debug-mode)
3. [Reading Ollama's response metadata](#3-reading-ollamas-response-metadata)
4. [Forward pass, prefill, and decode](#4-forward-pass-prefill-and-decode)
5. [Statelessness: who resends the history?](#5-statelessness-who-resends-the-history)
6. [The tool-calling protocol](#6-the-tool-calling-protocol)
7. [Tool calls across LLM providers](#7-tool-calls-across-llm-providers)

---

## 1. Environment & tooling (uv)

[uv](https://docs.astral.sh/uv/) is to Python roughly what `npm` + `nvm` +
`venv` are to Node, combined into one fast tool.

| Concept | npm | uv |
|---|---|---|
| Manifest | `package.json` | `pyproject.toml` |
| Lockfile | `package-lock.json` | `uv.lock` |
| Deps folder | `node_modules/` | `.venv/` |
| Add a dep | `npm install <pkg>` | `uv add <pkg>` |
| Run | `npm run <script>` | `uv run <cmd>` |

`uv init` is like `npm init` — it scaffolds a project — but it also pins a
Python version and manages the virtual environment for you (npm assumes Node is
already installed).

Install deps and run a script:

```bash
uv venv
uv pip install -r requirements.txt
uv run python 01_chat_loop.py   # uv run uses .venv automatically
```

---

## 2. Debug mode

All scripts share `debug_utils.py`. Toggle it per run with an env var:

```bash
DEBUG=1 uv run python 02_single_tool.py
```

In debug mode you see the full round trip as labeled blocks:

```
REQUEST           -> what we send to the model
RESPONSE          -> what the model sends back (incl. tool_calls)
TOOL EXECUTION    -> the function OUR code ran between model calls
REQUEST / RESPONSE-> the follow-up call that uses the tool result
```

Key implementation lesson: Ollama returns **pydantic objects**, not plain
dicts. Naively dumping them with `json.dumps(..., default=str)` collapses them
into a flat `repr()` string that hides nested structure (like the `tool_calls`
array inside the assistant message). The fix is a `default` hook that calls
`model_dump(mode="json")` to expand them into real nested JSON.

---

## 3. Reading Ollama's response metadata

A non-streaming response ends with timing/token fields. Durations are in
**nanoseconds** (divide by 1e9 for seconds).

| Field | Meaning |
|---|---|
| `created_at` | When generation finished (UTC, ISO 8601) |
| `done` | Whether generation is complete (`false` for streaming chunks) |
| `done_reason` | Why it stopped: `stop` (natural / tool call), `length` (hit limit), `load` |
| `total_duration` | End-to-end wall-clock time |
| `load_duration` | Time to get the model ready for this request |
| `prompt_eval_count` | **Input** tokens processed (system + history) |
| `prompt_eval_duration` | Time to ingest the prompt (prefill) |
| `eval_count` | **Output** tokens generated |
| `eval_duration` | Time spent generating (decode) |

Useful derived metric — generation speed:

```
eval_count / eval_duration(s)  ->  e.g. 994 / 20.59 ≈ 48 tokens/sec
```

### Why `load_duration` appears on *every* call

It's never truly zero. The expensive part (reading weights from disk into
RAM/VRAM) only happens on the **first** call (cold). Later calls are "warm" but
still report small per-request setup overhead (~100 ms).

Caveat: the model only stays warm for the **`keep_alive`** window (default
**5 minutes**). After that it unloads and the next call pays the cold cost
again. (`keep_alive: -1` keeps it loaded forever; `0` unloads immediately.)

### Why `prompt_eval_*` and `eval_*` instead of "input"/"output"

The names come from llama.cpp: running tokens through the network is
"evaluating" it (a forward pass). The distinction isn't input vs output, it's
**which kind of pass** — see the next section.

---

## 4. Forward pass, prefill, and decode

**Forward pass** = one execution of the model's math function: tokens in →
flow through every layer → out comes a prediction for the **next** token (a
probability distribution over the vocabulary). At inference time there are only
forward passes (the *backward* pass is for training).

Crucial fact: **one forward pass produces exactly one new token's prediction**,
no matter how many tokens you put in.

Generation therefore has two phases:

- **Prefill (`prompt_eval_*`)** — the model processes **all prompt tokens in a
  single batched forward pass** to build up its internal state (the KV cache),
  and emits the **first** output token. All prompt tokens are already known, so
  this is highly parallel and fast per token.
- **Decode (`eval_*`)** — the model generates **one token per forward pass**,
  feeding each new token back in, until it emits a stop token. Each token
  depends on the previous one, so this is **sequential** and slow per token.

This explains the ~10× speed gap we measured:

```
prefill: 61 tokens   in 0.139 s  ->  ~440 tok/s  (batched)
decode:  994 tokens  in 20.59 s  ->  ~48 tok/s   (one-by-one)
```

### The KV cache

During prefill the model computes **keys** and **values** for every prompt
token and caches them. During decode, each new token reuses that cache instead
of reprocessing the whole prompt every pass — turning what would be quadratic
work into a one-time prefill cost.

### Refined mental model

> Prefill ingests the whole prompt in one pass and emits the first token; then
> each following forward pass appends one more token, reusing the KV cache,
> until a stop token is produced.

Subtlety: a forward pass technically predicts a next token at *every* position,
but during prefill only the **last position's** prediction is kept (that's the
first output token). The other positions' predictions are only useful during
training. Consequence: time-to-first-token ≈ prefill time; time between later
tokens ≈ one decode pass.

---

## 5. Statelessness: who resends the history?

The model has **no memory**. Any "memory" you perceive is an illusion created
by a client **resending the entire conversation** on every call.

- In our scripts, *we* are that client: we maintain the `messages` list and
  resend it each turn (see `01_chat_loop.py`).
- The Ollama **CLI** (`ollama run`) and **GUI** do the same thing for you —
  they keep an in-memory history for the session and resend it via the same
  stateless `/api/chat` endpoint. `ollama run` is essentially `01_chat_loop.py`.

Proof: tell `ollama run` your name, quit, restart → it has forgotten, because
the client threw away its list. (This is what `05_persistent_memory.py` fixes.)

Nuance — **server-side KV cache ≠ memory**: if you resend a conversation that
shares a prefix with a previous one, the server can skip recomputing the
unchanged prefix. That's a *performance* cache (tied to `keep_alive` and the
model staying loaded), not stored conversation. The client still *sends* the
whole history; the server just chooses not to recompute the old part.

This is also why hosted APIs (OpenAI, Anthropic, ...) charge for input tokens
**every** turn — the whole history is re-sent each time.

---

## 6. The tool-calling protocol

Tool calling is not magic — it's a structured contract. The model can never
execute anything; it can only *ask*. Your code does the work.

The full cycle for one tool-using turn (two model calls):

```
1. REQUEST  -> messages + tools schema
2. RESPONSE -> assistant message with content="" and a tool_calls array
                 (the model ASKS; tool_calls live INSIDE the assistant message)
3. TOOL EXECUTION -> YOUR code looks up the function and runs it
4. append result as a role:"tool" message
5. REQUEST  -> whole history resent, now including the tool result
6. RESPONSE -> assistant message with the final text (tool_calls = null)
```

Things that surprised me:

- **`content` is empty on a tool-calling turn.** The model's entire output that
  turn *is* the `tool_calls`; the text answer only comes after it sees the
  result. (`thinking` is separate reasoning, not user-facing `content`.)
- **`tool_calls` is an array** because the model can request several tools at
  once — hence the `for call in tool_calls:` loop.
- **It takes two model calls** for one tool-using question. `02` hardcodes
  exactly one tool round; `04_agent_loop.py` generalizes this to "keep calling
  until the model returns a message with no `tool_calls`."
- **`done_reason: "stop"`** in round 1 means ending the turn with a tool call is
  a *normal* completion, not an error.
- **`prompt_eval_count` jumps** between the two calls (e.g. 200 → 1123) because
  round 2 resends everything *plus* the tool output — statelessness, visible in
  the numbers.

The protocol falls directly out of: *stateless model + client resends history +
model can only ask, never do.*

---

## 7. Tool calls across LLM providers

The **concept is universal**: the tool call is part of the assistant turn, you
execute it, and you feed the result back as a new message. Only the JSON
envelope differs. The parameter schema itself is plain **JSON Schema**
everywhere — only the wrapper and result-linking differ.

| | OpenAI / Ollama | Anthropic | Gemini |
|---|---|---|---|
| Where the call lives | `tool_calls` array (separate field) | `tool_use` block in `content[]` | `functionCall` in `parts[]` |
| Args type | JSON **string** | object (`input`) | object (`args`) |
| Assistant role name | `assistant` | `assistant` | `model` |
| Result carrier | message `role:"tool"` | `tool_result` block in a `user` msg | `functionResponse` part |
| Links result by | `tool_call_id` | `tool_use_id` | function **name** |
| Tool def wrapper | `function:{...}` + `parameters` | flat + `input_schema` | `functionDeclarations[]` + `parameters` |

### OpenAI / Ollama

```json
// assistant message (the call)
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": { "name": "read_file", "arguments": "{\"path\": \"~/notes.txt\"}" }
    }
  ]
}
// result you append
{ "role": "tool", "tool_call_id": "call_abc123", "content": "...file contents..." }
```

`arguments` is a JSON **string** (hence `02`'s `isinstance(fn_args, str)`
guard). Ollama often gives it already-parsed and omits the `id`.

### Anthropic (Claude)

```json
// assistant message (call is a block inside content; input is an object)
{
  "role": "assistant",
  "content": [
    { "type": "text", "text": "I'll read that file." },
    { "type": "tool_use", "id": "toolu_abc123", "name": "read_file", "input": { "path": "~/notes.txt" } }
  ]
}
// result goes back as a tool_result block inside a USER message
{
  "role": "user",
  "content": [
    { "type": "tool_result", "tool_use_id": "toolu_abc123", "content": "...file contents..." }
  ]
}
```

### Google Gemini

```json
// assistant turn (role is "model"); args is an object
{
  "role": "model",
  "parts": [
    { "functionCall": { "name": "read_file", "args": { "path": "~/notes.txt" } } }
  ]
}
// result part, linked by NAME (not an id)
{
  "role": "user",
  "parts": [
    { "functionResponse": { "name": "read_file", "response": { "content": "...file contents..." } } }
  ]
}
```

### Porting gotchas

- **Args: string vs object.** OpenAI/Ollama → JSON string; Anthropic/Gemini →
  parsed object.
- **Result linking:** `role:"tool"` + `tool_call_id` vs `tool_result` block +
  `tool_use_id` vs `functionResponse` by name.

This per-provider divergence is exactly what frameworks like **Smolagents** and
**LangGraph** (rungs `08`/`09`) normalize away — they're not magic, just
adapters over this same assistant-message-owns-the-tool-call pattern.

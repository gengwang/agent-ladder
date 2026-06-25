# LEARNING.md

A concept companion to the `agent-ladder` scripts. It explains the **LLM and
agent-harness ideas** behind what each rung demonstrates. For setup, debug
mode, and how to run scripts, see [`README.md`](./README.md).

## Table of contents

1. [Reading Ollama's response metadata](#1-reading-ollamas-response-metadata)
2. [Forward pass, prefill, and decode](#2-forward-pass-prefill-and-decode)
3. [Statelessness: who resends the history?](#3-statelessness-who-resends-the-history)
4. [The tool-calling protocol](#4-the-tool-calling-protocol)
5. [Tool calls across LLM providers](#5-tool-calls-across-llm-providers)
6. [Tool selection / routing with many tools](#6-tool-selection--routing-with-many-tools)
7. [Where tools run: custom vs built-in](#7-where-tools-run-custom-vs-built-in)
8. [The agent loop: looping until done](#8-the-agent-loop-looping-until-done)
9. [Persistent memory across script runs](#9-persistent-memory-across-script-runs)

---

## 1. Reading Ollama's response metadata

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

## 2. Forward pass, prefill, and decode

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

This explains the ~10× speed gap between the two phases:

```
prefill: 61 tokens   in 0.139 s  ->  ~440 tok/s  (batched)
decode:  994 tokens  in 20.59 s  ->  ~48 tok/s   (one-by-one)
```

### The KV cache

During prefill the model computes **keys** and **values** for every prompt
token and caches them. During decode, each new token reuses that cache instead
of reprocessing the whole prompt every pass — turning what would be quadratic
work into a one-time prefill cost.

### Mental model

> Prefill ingests the whole prompt in one pass and emits the first token; then
> each following forward pass appends one more token, reusing the KV cache,
> until a stop token is produced.

Subtlety: a forward pass technically predicts a next token at *every* position,
but during prefill only the **last position's** prediction is kept (that's the
first output token). The other positions' predictions are only useful during
training. Consequence: time-to-first-token ≈ prefill time; time between later
tokens ≈ one decode pass.

---

## 3. Statelessness: who resends the history?

The model has **no memory**. Any "memory" you perceive is an illusion created
by a client **resending the entire conversation** on every call.

- In these scripts, the script *is* that client: it maintains the `messages`
  list and resends it each turn (see `01_chat_loop.py`).
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

## 4. The tool-calling protocol

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

Key points:

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

### How the `tools` schema reaches the model

The `tools` array is a hand-written list, not something derived from your
functions or registered with the model:

- **Written by hand.** Each tool is a dict (`name`/`description`/`parameters`),
  collected into one list (`TOOLS`). A separate `AVAILABLE_FUNCTIONS` dict maps
  each `name` back to the real Python function.
- **Sent via `tools=`.** `ollama.chat(messages=..., tools=TOOLS)` serializes it
  into the request body. No kwarg, no tools — the model can't see them.
- **Resent every call**, exactly like `messages`. The model keeps no memory of
  the menu, so the whole schema rides along each turn (and counts toward
  `prompt_eval_count`).

### Schema = capability; prompt = policy

It's tempting to think the system prompt is what "tells" the model about a tool.
It isn't. Two different channels are at work:

- The **`tools` schema** is the capability — the real registration (name,
  parameters, JSON contract). ollama/llama.cpp renders it into the prompt as
  tokens for you, usually injected into the system section by the model's chat
  template.
- A sentence like "you have a read_file tool, use it when…" is just **policy**:
  a hint about *when* to reach for it.

Easy to prove by dropping one side: keep `tools=` but delete the sentence and the
model can still call the tool; keep the sentence but drop `tools=` and it can't —
at best it hallucinates a call your code never receives.

### Where tool policy belongs: system vs user

That policy is a message you write, so where it goes matters — put it in the
**system** message, not a user turn:

| | System message | User message |
|---|---|---|
| Scope | Standing rule for the whole conversation | Tied to one turn |
| Priority | Trained to outrank user instructions | Lower; a later user turn can override it |
| Drift | Stays pinned at the top each turn | Blurs into the task and washes out over turns |

That priority is a tendency from training, not a guarantee — small local models
like `qwen3:8b` honor it less reliably than frontier ones, so the system prompt
isn't a security boundary.

### The menu can change per turn

Since `tools` is resent every request, it doesn't have to stay the same. `02`/`03`
send a constant `TOOLS` only for simplicity — the model knows nothing beyond the
tools in the request it's looking at right now. Real harnesses lean on this:

- **Context-dependent menus.** Offer `write_file`/`delete_file` only once the
  user is in an edit mode, and hide them otherwise.
- **Gate by state.** Don't expose `deploy()` until tests pass; add it to the
  array only once that's true.
- **Phased workflows.** A planning turn gets read-only tools; an execution turn
  swaps in the mutating ones.
- **Token savings.** Schemas count toward `prompt_eval_count` every turn, so
  trimming the menu cuts input tokens — worth it once there are dozens of tools.
- **Forcing behavior.** Send a single tool (or none) to steer toward a specific
  action or force a plain-text answer.

Two things to watch when the menu shifts:

- **History references can dangle.** If turn 1 asked for `delete_file` and you
  drop it from turn 2's menu, that call and its `role:"tool"` result still live
  in `messages` and still feed back fine — the model just can't call it again.
  Usually harmless, but yanking tools mid-flow can confuse weaker models, since
  the history points at something no longer on offer.
- **Keep the dispatch map in sync.** `AVAILABLE_FUNCTIONS` is what actually runs
  a call. Anything advertised in `tools` should be runnable; anything the model
  names that isn't in the map falls through to the `ERROR: unknown tool ...` text
  result rather than crashing.

---

## 5. Tool calls across LLM providers

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

---

## 6. Tool selection / routing with many tools

With several tools (`03`), **routing is the model's job, not the harness's**.
There is no `if "file" in user_input: read_file()`. Three tools (`read_file`,
`list_dir`, `get_current_time`) are described, and the model decides — each
turn — whether to use a tool, *which* one, and with what arguments.

Key point:

- **The dispatch loop does not change between `02` and `03`.** It already
  iterates over `tool_calls` and looks each name up in a dict
  (`AVAILABLE_FUNCTIONS`). Adding tools is "grow the menu" (function + schema
  entry + one dict line); the plumbing is generic. The intelligence that grows
  lives in the **model**, not the harness code.

### Descriptions are the routing API

With one tool there was nothing to confuse. With several, the **tool
descriptions become the contract the model routes on.** `read_file` and
`list_dir` both take a `path` and both touch the filesystem — the only thing
that lets the model tell "read this file" from "what's in this folder" apart is
the prose in `description`. Vague descriptions → mis-routing. So the
`description` field is not documentation for humans; it's part of the model's
decision input, sent on every call inside `tools`.

### Two practical notes

- **No-arg tools** still need a schema: `"parameters": {"type": "object",
  "properties": {}}`. `get_current_time` also demonstrates a tool that needs
  no user-supplied data at all — pure "the model can't know this itself."
- **Correlating results** when several tools fire at once: the `role:"tool"`
  message includes `name`. OpenAI/Ollama formally link a result to its call by
  `tool_call_id` (see §5), but Ollama often omits the id, so including the tool
  `name` helps the model match each result to what it asked for.

### Mental model

> One tool taught the *protocol* (how a call/result round works). Many tools
> teach *routing* (the model choosing among options). The harness stays the
> same generic loop; what changes is that the menu — and therefore the quality
> of your descriptions — now decides how well the model performs.

---

## 7. Where tools run: custom vs built-in

The ladder builds *custom* tools: you author the schema, the model asks, your
code executes and feeds the result back. Real products also ship *built-in*
tools (file search, web search, code interpreter). The contract never changes —
the model can still only *ask*. What changes is **who sits on the "execute it
and feed back the result" side of the loop.** That ownership gives three
flavors:

| | Who writes schema | Who executes | Who appends result | Example |
|---|---|---|---|---|
| Custom (client-side) | You | You | You | `read_file` in `02`/`03` |
| Provider-defined, client-executed | Provider | You | You | Anthropic `text_editor` |
| Built-in / hosted (server-side) | Provider | Provider | Provider (hidden) | OpenAI `web_search` |

Ollama only supports the first row — there are no hosted tools locally, so every
tool in this repo is a custom tool.

### Concrete flow: provider-defined, client-executed

Anthropic's `text_editor` tool shows the middle row in action. You declare it by
*reference* — type + name, no `input_schema` — because the provider owns its
shape and the model was trained on it:

```json
"tools": [ { "type": "text_editor_20250124", "name": "str_replace_editor" } ]
```

From there the loop is identical to §4 — the model just speaks a command
vocabulary (`view`, `str_replace`, `create`, `insert`, …) that you never
defined:

1. User: "fix the typo in config.py".
2. Model → `tool_use`: `{command: "view", path: "config.py"}`.
3. **You** read the real local file, append a `tool_result` with its contents.
4. Model → `tool_use`: `{command: "str_replace", old_str, new_str}`.
5. **You** edit the real file, append `tool_result: "OK"`.
6. Model → final text: "Fixed the typo."

Steps 2–6 are the same ask → execute → feed-back loop as a custom tool. What
differs:

- **Schema ownership.** One-line declaration vs hand-authoring `input_schema`;
  the provider owns the field/command names.
- **Training alignment.** The model was post-trained on this exact shape, so it
  emits valid calls far more reliably than against a tool you invented. That
  reliability *is* the product.
- **A vocabulary you must fully honor.** Your executor has to implement every
  command the spec allows — the model can emit any of them; you don't pick the
  set.

The tell that execution is still *yours*: you see the `tool_use`/`tool_result`
round-trip. With a hosted tool (row 3) those steps happen in the provider's
sandbox and never appear in your code — you'd see only step 6.

### What's different about built-in (server-side) tools

The same loop still runs; it just runs **inside the provider's backend**, hidden
from you:

- **The round-trip collapses into one call from your side.** You enable
  `web_search`; internally the server does ask → execute → feed result → answer,
  and you get back the final assistant message (plus maybe citations). The
  two-call dance from §4 still happens — you're just no longer the one doing it.
- **Statelessness still holds underneath.** The provider resends history and
  tool results across those internal hops exactly like the local scripts do; it's
  just behind their endpoint, and you're billed for those hidden tokens.
- **You trade control for convenience.** The sandbox, rate limits, and
  implementation are theirs. Custom tools run wherever your code runs (local
  files, your DB), and the provider only ever sees the string you hand back.

### They mix, and "system tools" is overloaded

One request can enable hosted built-ins *and* pass your own `tools`. The model
routes across the combined menu (§6); for a built-in the server executes, for
one of yours you get the `tool_calls` and execute it. The model can't tell them
apart — both are just menu entries.

Watch the naming: in a hosted API, "system tools" usually means the server-
executed kind (row 3). But in a *harness/product* like Cursor, built-in
`read_file` / `codebase_search` / terminal tools are "system" only in that the
product ships them by default — mechanically they're still row 1 (the harness
defines the schema and executes locally). "Built-in" can mean "the product
wired it up," not "the model provider runs it."

---

## 8. The agent loop: looping until done

Rungs `02`/`03` allow exactly **one** tool round per user turn: call the model,
run whatever it asks for, call once more, print. That's fine for a single
lookup, but it can't express tools that must run **in sequence**, where the next
call depends on the previous result — e.g. "what's in the most recently named
file in this folder?" needs `list_dir` first, *then* a `read_file` chosen from
what that returned.

The agent loop drops the round counting and loops on a **condition** instead:

```
keep (call model -> run requested tools -> feed results back)
until the model returns a message with NO tool_calls.
```

The whole idea rests on one fact: **a message with no `tool_calls` is the
model's "I'm done" signal.** As long as it keeps asking for tools, there's more
work; the first turn it answers in plain text, the task is finished. That single
rule turns a fixed exchange into an open-ended agent that chains as many steps
as the task needs.

### What changes from `03`

- **Loop, not a fixed second call.** The two-call dance becomes
  `for step in range(MAX_STEPS): call; if no tool_calls: break; run tools`.
- **One append site.** The assistant message is appended after *every* model
  call, tool-requesting or final — replacing `03`'s two separate append spots.
  Each loop pass grows `messages` by the assistant turn plus any tool results,
  and that growing history is what lets the model "remember" what it already did
  this turn (statelessness, again — see §3).
- **Termination needs a guard.** "No tool_calls" is the normal exit. But a
  confused model can keep asking forever, so the loop is bounded by `MAX_STEPS`.
  Hitting that bound is a *safety stop*, not success — making the loop genuinely
  resilient (retries, bad arguments, tool exceptions) is rung `07`.

### Why this is "the" agent loop

Everything above this rung was setup; this is the part people mean by
"agentic." A model that can call tools, see results, and decide its *next* call
on its own — repeating until satisfied — is the core that frameworks dress up.
Planning, multi-agent orchestration, and error recovery (later rungs) are all
layered on top of this same `while not done` skeleton.

---

## 9. Persistent memory across script runs

Rungs `01`–`04` hold the entire conversation in a Python `messages` list that
lives only in RAM. Quit the process and that list is discarded — the model
appears to "forget," even though the weights never changed. The model was
never remembering; the client was holding history and throwing it away on exit
(see §3).

Persistent memory is therefore **not** a model capability. It is the harness
choosing to **save** the client-side `messages` list somewhere durable and
**reload** it on the next run. In `05_persistent_memory.py` that store is a JSON
file (`.agent_memory.json` by default); production systems use SQLite, Redis,
vector databases, or hosted thread APIs — same idea, different backing store.

### What changes from `04`

Only two hooks, both around the list that already existed:

```
startup:  messages = fresh system prompt + conversation loaded from disk
each turn: after the agent loop finishes, save the conversation to disk
```

The agent loop, tools, and termination rule are unchanged. Every API call is
still stateless: the full message list is resent on each model call, exactly
as before. Persistence only means the conversation survives a process restart
instead of starting empty.

### The system prompt is owned by code, not the file

A subtlety worth getting right: persist the *conversation*, but re-inject the
*system prompt* fresh on every run. On load, drop any saved `system` turn and
prepend the current `DEFAULT_SYSTEM_MESSAGE`; on save, write everything except
`system`.

The naive version — save the whole list verbatim, including `messages[0]` —
works, but it quietly makes the system prompt a one-time seed. The first run
saves it to disk, and every later run loads that copy instead of the code's.
Edit the prompt afterward and nothing changes until the file is deleted (the
"stale system prompt" trap). Splitting ownership keeps the prompt authoritative
in code while the conversation is the only thing that persists.

### Mental model

> The `messages` list is the only memory. RAM vs disk is an implementation
> detail of where *you* keep that list between runs. Delete the file and
> amnesia returns instantly — proof the model never stored anything.

### Costs and trade-offs

- **`prompt_eval_count` keeps growing** across sessions. Longer saved history
  means more input tokens (and cost, on hosted APIs) every turn — persistence
  is not free.
- **No summarization yet.** This rung saves verbatim history. Very long
  sessions eventually hit context limits; compaction and retrieval are separate
  problems (later rungs / frameworks).

### When to save: per turn vs. per message

`05` saves once per completed user turn by rewriting the whole file. Simplest
possible policy; two weaknesses at scale:

- **Crash safety.** If the agent loop dies mid-turn (after several tool rounds
  but before the final answer), nothing was written yet, so that work is lost.
- **Write cost.** Rewriting the entire list each turn is O(n) and grows with the
  conversation.

Production systems instead **append each message the moment it exists** — user
message before the model call, each tool call and result as they happen, the
assistant reply on completion. This is O(1) per message (one row in a DB, not a
full rewrite) and makes an agent *resumable*: a crash reloads partial progress
instead of losing it. Writes usually go to a real store (Postgres, Redis,
object storage), often async so saving never blocks the reply.

One caution: re-running a tool to "refresh" is safe for **reads** (`read_file`)
but dangerous for **actions** (`charge_card`, `deploy`) — those need idempotency
keys so a retry doesn't fire the side effect twice.

### Tool results go stale, and nothing notices

A tool result is a `role:"tool"` string frozen into the history at the instant it
ran — a snapshot, not a live view. The file gets edited on disk; the cached
`read_file` result in `messages` does not change. Yet on resend it looks exactly
as authoritative as a fresh one.

The model cannot catch this, because of three facts that are easy to forget:

- **No sense of time.** Everything in the context window reads as "now." A result
  from 20 turns ago is indistinguishable from one from this turn.
- **No sense of change.** History is an append-only log of past observations;
  nothing rewrites an old result when the world moves on.
- **No ground truth to check against.** The model sees only tokens, never the
  world, so it has nothing to compare the snapshot to — and it states a stale
  value as fluently as a live one. It doesn't get it wrong *and notice*; it
  can't notice at all.

So freshness is never the model's job — it's something the harness/platform must
engineer: re-fetch volatile data instead of trusting history, stamp results with
a timestamp, expire cached results with a TTL, track versions/ETags, or prune
old results so they can't mislead. A system-prompt rule ("re-read before
answering") nudges the model to re-call, but it's a tendency, not a guarantee.

### Harness vs. platform

These two words name different layers, and persistence/freshness split across
both:

- **Harness** = the agent loop itself: assemble `messages`, call the model, run
  tools, feed results back, loop until done. Each `NN_*.py` script *is* a
  harness; Smolagents/LangGraph (rungs `08`/`09`) are harness frameworks.
- **Platform** = the infrastructure around it that runs harnesses as a service:
  storage, auth, multi-user sessions, model serving, billing, monitoring.

A platform runs one or more harnesses — so the harness is best seen as the
specialized *agent-runtime* layer of a platform, while the rest is generic
infrastructure any harness could plug into. (A harness can also run with no
platform at all — that's exactly what these scripts are.)

Mapped to this rung: deciding *whether to re-fetch*, pruning context, and the
system-prompt policy are **harness** choices; the backing store, TTL caches, and
version tracking are **platform** choices.

### The big picture

The model is a pure reasoning function over the tokens it's handed. It has no
memory, no clock, and no window onto the world — every way it *seems* stateful or
time-aware is scaffolding the harness bolted on: resending history (§3), saving
it across runs (this rung), and re-grounding stale facts. The rest of the ladder
— orchestration (`06`), resilience (`07`), frameworks (`08`/`09`) — is all
answers to the same question: *how do we compensate for what the model
fundamentally can't know on its own?*

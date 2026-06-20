"""
01_chat_loop.py

The most basic possible "agent": a plain chat loop, no tools.

CONCEPT TO INTERNALIZE:
The model is stateless. It has no memory of anything you didn't just send it.
Every single call, we resend the ENTIRE conversation history so far. The
"memory" you perceive in a chat app is an illusion created by the client
(this script) re-sending everything each time, not by the model remembering.

Try this experiment once it's running:
  1. Tell it your name.
  2. Ask "what's my name?" -> it works, because history was resent.
  3. Comment out the `messages.append(...)` line that saves history,
     restart, and try again -> it will have no idea, even mid-conversation.
"""

import json
import os

import ollama

MODEL = "qwen3:8b"  # change to whatever you have pulled, e.g. "llama3.1:8b"

# Flip on by running:  DEBUG=1 uv run python 01_chat_loop.py
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


def _dump(label, payload):
    """Pretty-print a request/response payload when DEBUG is enabled."""
    print(f"\n----- {label} -----")
    print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    print("-" * (12 + len(label)) + "\n")


def main():
    # This list IS the entire memory of the conversation.
    # Nothing exists for the model outside of what's in this list.
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."}
    ]

    print(f"Chat loop running with model={MODEL}. Type 'exit' to quit.")
    if DEBUG:
        print("DEBUG mode ON: printing full request/response payloads.")
    print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "/bye"):
            break

        # 1. Append the user's message to the running history
        messages.append({"role": "user", "content": user_input})

        # 2. Send the WHOLE history to the model. Every. Single. Time.
        if DEBUG:
            _dump("REQUEST", {"model": MODEL, "messages": messages})

        response = ollama.chat(model=MODEL, messages=messages)

        if DEBUG:
            # response is an ollama ChatResponse object; dict() shows everything
            # the server returned (timings, token counts, the message, etc.).
            _dump("RESPONSE", dict(response))

        assistant_message = response["message"]["content"]
        print(f"Assistant: {assistant_message}\n")

        # 3. Append the assistant's reply too, so the next call includes it.
        #    Comment this line out to see the "amnesia" effect described above.
        messages.append({"role": "assistant", "content": assistant_message})


if __name__ == "__main__":
    main()

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

import os

import ollama

import cli_utils
import debug_utils

MODEL = os.environ.get("MODEL", "qwen3:8b")


def main():
    # This list IS the entire memory of the conversation.
    # Nothing exists for the model outside of what's in this list.
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."}
    ]

    print(f"Chat loop running with model={MODEL}. Type 'exit' to quit.")
    debug_utils.banner()
    print()

    while True:
        user_input = input("You: ").strip()
        if cli_utils.is_exit_command(user_input):
            break

        # 1. Append the user's message to the running history
        messages.append({"role": "user", "content": user_input})

        # 2. Send the WHOLE history to the model. Every. Single. Time.
        debug_utils.dump_request(MODEL, messages)

        response = ollama.chat(model=MODEL, messages=messages)

        debug_utils.dump_response(response)

        assistant_message = response["message"]["content"]
        print(f"Assistant: {assistant_message}\n")

        # 3. Append the assistant's reply too, so the next call includes it.
        #    Comment this line out to see the "amnesia" effect described above.
        messages.append({"role": "assistant", "content": assistant_message})


if __name__ == "__main__":
    main()

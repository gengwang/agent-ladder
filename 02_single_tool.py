"""
02_single_tool.py

Add exactly ONE tool to the chat loop: read_file(path).

CONCEPT TO INTERNALIZE:
"Tool calling" is not magic. It's a structured contract:
  1. You describe the tool to the model as a JSON schema (name, description,
     parameters) -- you are NOT giving the model code, just a description.
  2. The model, instead of replying with text, can reply with a structured
     request: "call read_file with path=foo.txt".
  3. YOU are the one who actually runs the function. The model never
     executes anything itself -- it only ever asks you to.
  4. You take the function's return value, feed it back into the
     conversation as a new message, and call the model again so it can
     use that result to form its final answer.

This is the entire trick behind every "agentic" framework. Everything from
here on is just adding more tools, more loop iterations, and more state
around this exact same pattern.
"""

import json
import os
import ollama

MODEL = "qwen3:8b"


# ---- Step A: the actual Python function the tool will run ----
def read_file(path: str) -> str:
    """Reads and returns the contents of a text file."""
    try:
        with open(os.path.expanduser(path), "r") as f:
            return f.read()
    except Exception as e:
        # Tool errors should come back as text the MODEL can read and
        # react to, not as a Python exception that crashes your script.
        return f"ERROR reading file: {e}"


# ---- Step B: describe that function to the model in its expected schema ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file from disk.",
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
    }
]

# Map tool name -> actual Python function. This is how we go from
# "the model asked for 'read_file'" to "run this real function".
AVAILABLE_FUNCTIONS = {
    "read_file": read_file,
}


def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. You have a read_file tool. "
                "Use it when the user asks about a file's contents."
            ),
        }
    ]

    print(f"Single-tool agent running with model={MODEL}. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        # First call: give the model the chance to request a tool call.
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        message = response["message"]

        tool_calls = message.get("tool_calls")

        if tool_calls:
            # The model wants to use one or more tools before answering.
            # Always append the assistant's tool-call message to history
            # first, so the model can later see what IT asked for.
            messages.append(message)

            for call in tool_calls:
                fn_name = call["function"]["name"]
                fn_args = call["function"]["arguments"]
                # Ollama may give args as a dict already, or as a JSON string
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)

                print(f"  [tool call] {fn_name}({fn_args})")

                fn = AVAILABLE_FUNCTIONS.get(fn_name)
                result = fn(**fn_args) if fn else f"ERROR: unknown tool {fn_name}"

                # Feed the tool's result back in as a 'tool' role message.
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                    }
                )

            # Second call: now the model has the tool result and can
            # form its actual answer to the user.
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            message = response["message"]

        assistant_text = message.get("content", "")
        print(f"Assistant: {assistant_text}\n")
        messages.append(message)


if __name__ == "__main__":
    main()

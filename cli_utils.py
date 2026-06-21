"""
cli_utils.py

Shared helpers for the interactive REPL shell that every rung wraps around.

`debug_utils` centralizes how we talk to the MODEL; this module centralizes how
we talk to the USER at the prompt. Both exist for the same reason: behavior that
should be identical across every script lives in exactly one place, so a fix or
tweak (here, "what counts as 'quit'?") lands everywhere at once instead of
drifting per file.
"""

# The single source of truth for "the user wants to leave". Defined once so
# every rung accepts the same set of exit words -- no more 01 knowing "/bye"
# while 02 and 03 silently don't.
EXIT_COMMANDS = ("exit", "quit", "/bye")


def is_exit_command(user_input: str) -> bool:
    """True if the user's input is a request to end the session.

    Matching is case-insensitive and ignores surrounding whitespace, so "Exit",
    " QUIT ", and "/bye" all count.
    """
    return user_input.strip().lower() in EXIT_COMMANDS

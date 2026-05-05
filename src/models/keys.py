from getpass import getpass
import os
import sys


def resolve_api_key(
    explicit_key: str | None,
    env_var: str,
    label: str,
) -> str:
    if explicit_key:
        return explicit_key

    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    if sys.stdin.isatty():
        entered = getpass(f"{label} not set. Enter {env_var}: ").strip()
        if entered:
            return entered

    raise ValueError(
        f"{env_var} is not set. Export it in your environment or pass api_key explicitly."
    )
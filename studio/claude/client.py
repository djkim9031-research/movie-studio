"""The single path for Claude API calls: route -> model and effort, spend caps before, ledger after."""

import os
from pathlib import Path

import anthropic

from studio.claude.ledger import Ledger, hops_from_message
from studio.config import Config

API_URL = "https://api.anthropic.com"


class CredentialsError(RuntimeError):
    pass


class Refused(RuntimeError):
    pass


def read_api_key(env_file: Path) -> str:
    """Reads ANTHROPIC_API_KEY from a KEY=VALUE file that only its owner can read."""
    env_file = Path(env_file)
    if not env_file.exists():
        raise CredentialsError(f"{env_file} is missing; put ANTHROPIC_API_KEY=... in it with mode 0600")
    if env_file.stat().st_mode & 0o077:
        raise CredentialsError(f"{env_file} must be mode 0600")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().removeprefix("export ").partition("=")
        if key.strip() == "ANTHROPIC_API_KEY" and value.strip():
            return value.strip().strip("'\"")
    raise CredentialsError(f"no ANTHROPIC_API_KEY in {env_file}")


class Claude:
    def __init__(self, cfg: Config, *, http_client=None, ledger: Ledger | None = None):
        # The SDK adds this variable's headers to every request whatever the client is given.
        if os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
            raise CredentialsError("unset ANTHROPIC_CUSTOM_HEADERS; studio sends no extra headers")
        kwargs = {"http_client": http_client} if http_client is not None else {}
        # An explicit base_url keeps ANTHROPIC_BASE_URL from sending the key and prompts to another host.
        self.client = anthropic.Anthropic(api_key=read_api_key(cfg.anthropic_env), base_url=API_URL, **kwargs)
        self.cfg = cfg
        self.ledger = ledger or Ledger(cfg.ledger_dir, cfg.claude)

    def create(self, route: str, *, film: str, **params):
        target = self.cfg.claude.routes[route]
        self.ledger.check_caps(film)
        output_config = {"effort": target.effort, **params.pop("output_config", {})}
        message = self.client.messages.create(model=target.model, output_config=output_config, **params)
        request_id = getattr(message, "_request_id", None)
        try:
            self.ledger.record(route=route, film=film, hops=hops_from_message(message, target.model),
                               request_id=request_id)
        except Exception:
            # The call is already paid for; keep a row at the highest rates before the error surfaces.
            self.ledger.record_unpriced(route=route, film=film, usage=message.usage, request_id=request_id)
            raise
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise Refused(f"{route}: refused ({getattr(details, 'category', None)})")
        return message

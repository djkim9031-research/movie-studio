"""The single path for Claude API calls: route -> model and effort, spend caps before, ledger after."""

import os
from pathlib import Path

import anthropic

from studio.claude.ledger import Ledger, hops_from_message
from studio.config import Config
from studio.secrets import SecretError, read_secret

API_URL = "https://api.anthropic.com"

CredentialsError = SecretError


class Refused(RuntimeError):
    pass


def read_api_key(env_file: Path) -> str:
    return read_secret(env_file, "ANTHROPIC_API_KEY")


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
        # Streamed: the SDK refuses a plain request whose max_tokens could take more than 10 minutes.
        with self.client.messages.stream(model=target.model, output_config=output_config, **params) as stream:
            message = stream.get_final_message()
        request_id = stream.request_id
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

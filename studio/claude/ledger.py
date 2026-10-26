"""Spend ledger for Claude API calls: one JSON line per call, one file per month."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from studio.config import ClaudeConfig

MILLION = 1_000_000


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Hop:
    """Tokens one model billed within a call; a fallback call has one hop per model that ran."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0


def price_hop(hop: Hop, cfg: ClaudeConfig) -> tuple[float, bool]:
    """Returns (USD, priced); unknown models are charged at cfg.unknown_price and flagged."""
    price = cfg.prices.get(hop.model)
    known = price is not None
    price = price or cfg.unknown_price
    usd = (hop.input_tokens * price.input
           + hop.output_tokens * price.output
           + hop.cache_read * price.cache_read
           + hop.cache_write_5m * price.input * cfg.cache_write_5m
           + hop.cache_write_1h * price.input * cfg.cache_write_1h) / MILLION
    return usd, known


def _hop(model: str, usage) -> Hop:
    written = getattr(usage, "cache_creation_input_tokens", None) or 0
    by_ttl = getattr(usage, "cache_creation", None)
    return Hop(model=model,
               input_tokens=usage.input_tokens or 0,
               output_tokens=usage.output_tokens or 0,
               cache_read=getattr(usage, "cache_read_input_tokens", None) or 0,
               cache_write_5m=by_ttl.ephemeral_5m_input_tokens if by_ttl else written,
               cache_write_1h=by_ttl.ephemeral_1h_input_tokens if by_ttl else 0)


def hops_from_message(message, requested_model: str) -> list[Hop]:
    """One hop per model that billed. With iterations (fallbacks, compaction) the top-level usage is their
    sum, so only the iterations are priced; an iteration without a model ran on the requested one."""
    iterations = getattr(message.usage, "iterations", None)
    if iterations:
        return [_hop(str(getattr(it, "model", None) or requested_model), it) for it in iterations]
    return [_hop(str(message.model), message.usage)]


class Ledger:
    def __init__(self, directory: Path, cfg: ClaudeConfig, clock=datetime.now):
        self.directory = Path(directory)
        self.cfg = cfg
        self.clock = clock

    def record(self, *, route: str, film: str, hops: list[Hop], request_id: str | None = None) -> float:
        now = self.clock()
        priced = [(hop, *price_hop(hop, self.cfg)) for hop in hops]
        total = sum(usd for _, usd, _ in priced)
        row = {"ts": now.isoformat(timespec="seconds"), "route": route, "film": film, "request_id": request_id,
               "usd": round(total, 6),
               "hops": [{**asdict(hop), "usd": round(usd, 6), "priced": known} for hop, usd, known in priced]}
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.directory / f"{now:%Y-%m}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        return total

    def entries(self, month: str | None = None) -> list[dict]:
        files = [self.directory / f"{month}.jsonl"] if month else sorted(self.directory.glob("*.jsonl"))
        rows = []
        for path in files:
            if path.exists():
                rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return rows

    def month_total(self, month: str) -> float:
        return sum(row["usd"] for row in self.entries(month))

    def film_total(self, film: str) -> float:
        return sum(row["usd"] for row in self.entries() if row["film"] == film)

    def check_caps(self, film: str) -> None:
        month = f"{self.clock():%Y-%m}"
        spent_month = self.month_total(month)
        if spent_month >= self.cfg.month_cap_usd:
            raise BudgetExceeded(f"{month} spend ${spent_month:.2f} reached the ${self.cfg.month_cap_usd:.2f} cap")
        spent_film = self.film_total(film)
        if spent_film >= self.cfg.film_cap_usd:
            raise BudgetExceeded(f"film {film} spend ${spent_film:.2f} reached the ${self.cfg.film_cap_usd:.2f} cap")

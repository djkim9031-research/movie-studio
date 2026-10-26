"""Spend ledger for Claude API calls: one JSON line per call, one file per month."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from studio.config import ClaudeConfig
from studio.fs import append_private, private_dir

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


def _field(obj, name: str):
    """Usage arrives as SDK objects, or as plain dicts for fields the SDK type does not declare."""
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _tokens(obj, name: str) -> int:
    return int(_field(obj, name) or 0)


def _hop(model: str, usage) -> Hop:
    by_ttl = _field(usage, "cache_creation")
    five = _tokens(by_ttl, "ephemeral_5m_input_tokens")
    hour = _tokens(by_ttl, "ephemeral_1h_input_tokens")
    # Writes the TTL breakdown does not account for are charged at the dearer 1-hour rate.
    hour += max(_tokens(usage, "cache_creation_input_tokens") - five - hour, 0)
    return Hop(model=model,
               input_tokens=_tokens(usage, "input_tokens"),
               output_tokens=_tokens(usage, "output_tokens"),
               cache_read=_tokens(usage, "cache_read_input_tokens"),
               cache_write_5m=five,
               cache_write_1h=hour)


def hops_from_message(message, requested_model: str) -> list[Hop]:
    """One hop per model that billed. With iterations (fallbacks, compaction) every attempt is listed there and
    the top-level usage covers only the attempt that replied, so only the iterations are priced; an iteration
    without a model ran on the requested one."""
    iterations = _field(message.usage, "iterations")
    if iterations:
        return [_hop(str(_field(it, "model") or requested_model), it) for it in iterations]
    return [_hop(str(message.model), message.usage)]


def summarize(entries: list[dict]) -> dict:
    by_film: dict[str, float] = {}
    by_route: dict[str, float] = {}
    for row in entries:
        by_film[row["film"]] = by_film.get(row["film"], 0.0) + row["usd"]
        by_route[row["route"]] = by_route.get(row["route"], 0.0) + row["usd"]
    return {"total": sum(row["usd"] for row in entries),
            "calls": len(entries),
            "unpriced_calls": sum(1 for row in entries if not all(hop["priced"] for hop in row["hops"])),
            "by_film": by_film,
            "by_route": by_route}


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
        append_private(private_dir(self.directory) / f"{now:%Y-%m}.jsonl", json.dumps(row, sort_keys=True) + "\n")
        return total

    def record_unpriced(self, *, route: str, film: str, usage, request_id: str | None = None) -> float:
        """For a paid call whose usage could not be priced: top-level tokens at the unknown-model rates."""
        hop = Hop(model="unpriced",
                  input_tokens=_tokens(usage, "input_tokens"),
                  output_tokens=_tokens(usage, "output_tokens"),
                  cache_read=_tokens(usage, "cache_read_input_tokens"),
                  cache_write_1h=_tokens(usage, "cache_creation_input_tokens"))
        return self.record(route=route, film=film, hops=[hop], request_id=request_id)

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

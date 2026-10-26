from datetime import datetime

import pytest
from anthropic.types import Message
from anthropic.types.beta import BetaMessage

from studio.claude.ledger import BudgetExceeded, Hop, Ledger, hops_from_message, price_hop, summarize
from studio.config import ClaudeConfig, Price

PRICES = ClaudeConfig(
    prices={
        "claude-opus-5-5": Price(input=4.0, output=20.0, cache_read=0.2),
        "claude-opus-5": Price(input=5.0, output=25.0, cache_read=0.5),
    },
    film_cap_usd=2.0,
    month_cap_usd=3.0,
)
M = 1_000_000


def message_json(usage, model="claude-opus-5-5"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": usage}


def test_price_input_and_output():
    assert price_hop(Hop("claude-opus-5-5", input_tokens=M), PRICES) == (pytest.approx(4.0), True)
    assert price_hop(Hop("claude-opus-5-5", output_tokens=M), PRICES) == (pytest.approx(20.0), True)


def test_price_cache_reads_and_writes():
    assert price_hop(Hop("claude-opus-5-5", cache_read=M), PRICES)[0] == pytest.approx(0.2)
    assert price_hop(Hop("claude-opus-5-5", cache_write_5m=M), PRICES)[0] == pytest.approx(5.0)
    assert price_hop(Hop("claude-opus-5-5", cache_write_1h=M), PRICES)[0] == pytest.approx(8.0)


def test_unknown_model_uses_the_unknown_price_and_is_flagged():
    assert price_hop(Hop("claude-new", input_tokens=M), PRICES) == (pytest.approx(10.0), False)


def test_hops_from_plain_message_split_cache_writes_by_ttl():
    msg = Message.model_validate(message_json({
        "input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 3000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000}}))
    assert hops_from_message(msg, "claude-opus-5-5") == [
        Hop("claude-opus-5-5", input_tokens=100, output_tokens=50, cache_read=2000,
            cache_write_5m=1000, cache_write_1h=2000)]


def test_hops_from_fallback_price_each_iteration_at_its_model():
    msg = BetaMessage.model_validate(message_json(model="claude-opus-5", usage={
        "input_tokens": 300, "output_tokens": 90,
        "iterations": [
            {"type": "message", "input_tokens": 100, "output_tokens": 0,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 200, "output_tokens": 90,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}]}))
    assert hops_from_message(msg, "claude-opus-5-5") == [
        Hop("claude-opus-5-5", input_tokens=100), Hop("claude-opus-5", input_tokens=200, output_tokens=90)]


def make_ledger(tmp_path):
    return Ledger(tmp_path, PRICES, clock=lambda: datetime(2026, 10, 26, 12, 0))


def test_record_writes_a_monthly_file_and_sums_by_film(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record(route="doctor", film="a", hops=[Hop("claude-opus-5-5", input_tokens=M // 4)])
    ledger.record(route="doctor", film="b", hops=[Hop("claude-opus-5-5", output_tokens=M // 20)])
    assert (tmp_path / "2026-10.jsonl").exists()
    assert ledger.month_total("2026-10") == pytest.approx(2.0)
    assert ledger.film_total("a") == pytest.approx(1.0)


def test_caps_allow_spending_below_both_caps(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record(route="doctor", film="a", hops=[Hop("claude-opus-5-5", input_tokens=M // 4)])
    ledger.check_caps("a")


def test_caps_refuse_when_the_film_cap_is_reached(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record(route="doctor", film="a", hops=[Hop("claude-opus-5-5", input_tokens=M // 2)])
    with pytest.raises(BudgetExceeded, match="film a"):
        ledger.check_caps("a")


def test_caps_refuse_when_the_month_cap_is_reached(tmp_path):
    ledger = make_ledger(tmp_path)
    for film in ("a", "b", "c"):
        ledger.record(route="doctor", film=film, hops=[Hop("claude-opus-5-5", input_tokens=M // 4)])
    with pytest.raises(BudgetExceeded, match="2026-10"):
        ledger.check_caps("d")


def test_summary_totals_by_film_and_route_and_counts_unpriced_calls(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record(route="doctor", film="a", hops=[Hop("claude-opus-5-5", input_tokens=M // 4)])
    ledger.record(route="plan", film="a", hops=[Hop("claude-opus-5-5", output_tokens=M // 20)])
    ledger.record(route="plan", film="b", hops=[Hop("claude-new", input_tokens=M // 10)])
    summary = summarize(ledger.entries("2026-10"))
    assert summary["calls"] == 3 and summary["unpriced_calls"] == 1
    assert summary["total"] == pytest.approx(3.0)
    assert summary["by_film"] == {"a": pytest.approx(2.0), "b": pytest.approx(1.0)}
    assert summary["by_route"] == {"doctor": pytest.approx(1.0), "plan": pytest.approx(2.0)}


def test_ledger_files_are_owner_only(tmp_path):
    ledger = Ledger(tmp_path / "ledger", PRICES, clock=lambda: datetime(2031, 2, 3))
    ledger.record(route="doctor", film="a", hops=[Hop("claude-opus-5-5", input_tokens=1)])
    assert (tmp_path / "ledger").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "ledger" / "2031-02.jsonl").stat().st_mode & 0o777 == 0o600

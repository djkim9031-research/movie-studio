from datetime import datetime

from typer.testing import CliRunner

from studio.claude.ledger import Hop, Ledger
from studio.cli import app
from studio.config import load_config


def test_cost_prints_the_month_total_against_the_cap(tmp_path):
    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\n"
                        "claude:\n  month_cap_usd: 40\n"
                        "  prices:\n    claude-opus-5-5: {input: 4.0, output: 20.0, cache_read: 0.2}\n")
    cfg = load_config(cfg_file)
    Ledger(cfg.ledger_dir, cfg.claude, clock=lambda: datetime(2026, 10, 26)).record(
        route="doctor", film="_doctor", hops=[Hop("claude-opus-5-5", input_tokens=250_000)])
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "cost", "--month", "2026-10"])
    assert result.exit_code == 0, result.output
    assert "2026-10: $1.00 of $40.00 cap, 1 calls" in result.output
    assert "_doctor" in result.output

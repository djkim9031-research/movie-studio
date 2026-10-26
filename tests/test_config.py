from pathlib import Path

from studio.config import load_config


def test_data_root_expands_home(tmp_path):
    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text("data_root: ~/somewhere\n")
    assert load_config(cfg_file).data_root == Path.home() / "somewhere"


def test_paths_derive_from_data_root(tmp_path):
    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\n")
    cfg = load_config(cfg_file)
    assert cfg.ledger_dir == tmp_path / "ledger"
    assert cfg.bin_dir == tmp_path / "bin"
    assert cfg.anthropic_env == tmp_path / "secrets" / "anthropic.env"


def test_repo_config_routes_doctor_to_opus():
    cfg = load_config()
    assert cfg.claude.routes["doctor"].model == "claude-opus-5-5"
    assert "claude-opus-5-5" in cfg.claude.prices

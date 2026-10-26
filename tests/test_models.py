import hashlib
import os

import pytest

from studio import models
from studio.config import Config
from studio.secrets import SecretError, read_secret


def file_entry(root, rel, data):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def make_config(tmp_path, files):
    return Config.model_validate({
        "data_root": str(tmp_path),
        "models": {"demo": {"repo": "org/demo", "revision": "abc123", "files": files}}})


def test_verify_passes_when_every_file_matches(tmp_path):
    root = tmp_path / "models" / "comfy"
    cfg = make_config(tmp_path, [file_entry(root, "vae/a.safetensors", b"aaaa")])
    assert models.verify(cfg, "demo") == []


def test_verify_names_missing_short_and_altered_files(tmp_path):
    root = tmp_path / "models" / "comfy"
    good = file_entry(root, "vae/a.safetensors", b"aaaa")
    short = {**file_entry(root, "vae/b.safetensors", b"bb"), "size": 3}
    altered = {**file_entry(root, "vae/c.safetensors", b"cc"), "sha256": "0" * 64}
    missing = {"path": "vae/d.safetensors", "size": 1, "sha256": "0" * 64}
    problems = models.verify(make_config(tmp_path, [good, short, altered, missing]), "demo")
    assert [p.split(":")[0] for p in problems] == ["vae/b.safetensors", "vae/c.safetensors", "vae/d.safetensors"]
    assert "size" in problems[0] and "sha256" in problems[1] and "missing" in problems[2]


def test_fetch_downloads_each_file_at_the_pinned_revision_then_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.invalid")
    payload = b"weights"
    entry = {"path": "vae/a.safetensors", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    cfg = make_config(tmp_path, [entry])
    (tmp_path / "secrets").mkdir()
    token_file = tmp_path / "secrets" / "hf.env"
    token_file.write_text("HF_TOKEN=hf_test\n")
    os.chmod(token_file, 0o600)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = kwargs["local_dir"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    monkeypatch.setattr(models, "hf_hub_download", fake_download)
    assert models.fetch(cfg, "demo") == []
    assert calls == [{"repo_id": "org/demo", "filename": "vae/a.safetensors", "revision": "abc123",
                      "local_dir": cfg.models_dir, "token": "hf_test", "endpoint": "https://huggingface.co"}]


def test_read_secret_reads_the_named_key(tmp_path):
    path = tmp_path / "hf.env"
    path.write_text("HF_TOKEN=hf_abc\n")
    os.chmod(path, 0o600)
    assert read_secret(path, "HF_TOKEN") == "hf_abc"


def test_read_secret_refuses_a_file_others_can_read(tmp_path):
    path = tmp_path / "hf.env"
    path.write_text("HF_TOKEN=hf_abc\n")
    os.chmod(path, 0o644)
    with pytest.raises(SecretError, match="0600"):
        read_secret(path, "HF_TOKEN")


def test_a_set_with_its_own_root_lives_outside_comfyuis_model_folders(tmp_path):
    root = tmp_path / "models" / "faces"
    cfg = Config.model_validate({
        "data_root": str(tmp_path),
        "models": {"faces": {"repo": "org/faces", "revision": "abc123", "root": "faces",
                             "files": [file_entry(root, "detector.onnx", b"onnx")]}}})
    assert cfg.model_root("faces") == root
    assert models.verify(cfg, "faces") == []


def test_a_set_root_must_be_a_plain_folder_name(tmp_path):
    with pytest.raises(ValueError):
        Config.model_validate({"data_root": str(tmp_path),
                               "models": {"x": {"repo": "o/x", "revision": "r", "root": "../elsewhere", "files": []}}})


def test_present_checks_names_and_sizes_without_reading_the_files(tmp_path, monkeypatch):
    root = tmp_path / "models" / "comfy"
    good = file_entry(root, "vae/a.safetensors", b"aaaa")
    short = {**file_entry(root, "vae/b.safetensors", b"bb"), "size": 3}
    missing = {"path": "vae/d.safetensors", "size": 1, "sha256": "0" * 64}
    monkeypatch.setattr(models, "sha256_file", lambda path: pytest.fail("present must not hash"))
    problems = models.present(make_config(tmp_path, [good, short, missing]), "demo")
    assert problems == ["vae/b.safetensors: size 2, expected 3", "vae/d.safetensors: missing"]

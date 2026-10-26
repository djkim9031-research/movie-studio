import numpy as np
import pytest
import yaml
from PIL import Image

from studio import cast, identity, provenance
from studio.config import REPO_CONFIG, Config, load_config
from studio.media import ffmpeg
from tests.test_identity import make_photos
from tests.test_refs import ME, OTHER, FakeTools, face, unit

needs_ffmpeg = pytest.mark.skipif(not ffmpeg.binary(load_config()).exists(), reason="run: uv run studio fetch-ffmpeg")


@pytest.fixture
def cfg(tmp_path):
    repo = yaml.safe_load(REPO_CONFIG.read_text())
    (tmp_path / "bin").symlink_to(load_config().bin_dir)
    return Config.model_validate({"data_root": str(tmp_path), "models": repo["models"]})


def test_angle_words_name_the_size_of_the_turn_not_its_side():
    assert [cast.angle_words(y) for y in (0.0, -0.2, 0.25, 0.5, -0.7)] == [
        "frontal", "slightly turned", "slightly turned", "three-quarter view", "near profile"]
    assert cast.angle_bin(-0.7) == 0 and cast.angle_bin(0.0) == 3 and cast.angle_bin(0.7) == 6


def test_ingest_needs_a_consent_note(cfg):
    make_photos(cfg)
    with pytest.raises(ValueError, match="consent"):
        cast.ingest(cfg, "me", FakeTools({}), echo=print)


@needs_ffmpeg
def test_ingest_keeps_the_person_by_angle_without_near_copies_and_reports_missing_angles(cfg):
    _, tools = make_photos(cfg)
    identity.save_profile(cfg, identity.build_profile(cfg, "me", tools))
    root = cfg.data_root / "cast" / "me"
    (root / "consent.txt").write_text("me, 2026-10-26, films on this machine only\n")
    (root / "video").mkdir()
    ffmpeg.run(load_config(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=3",
               "-pix_fmt", "yuv420p", str(root / "video" / "turn.mp4")).check_returncode()
    turned = face(200, 60, 200, unit(1, 0.25, 0.25), yaw=0.45)
    tools.by_size[(360, 640)] = [turned, face(450, 60, 150, OTHER)]
    report = cast.ingest(cfg, "me", tools, echo=print, trigger="zqme")
    images = sorted((root / "dataset" / "images").glob("*.png"))
    assert len(images) == report["kept"] and report["kept"] >= 2
    assert report["bins"]["three-quarter view"] == 1
    assert "near profile" in report["missing"]
    captions = [p.with_suffix(".txt").read_text() for p in images]
    assert all(c.startswith("photo of zqme person") for c in captions)
    assert any("three-quarter view" in c for c in captions)
    assert all(provenance.problems(cfg, p) == [] for p in images)
    assert not (root / ".work").exists()

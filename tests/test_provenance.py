import json

import pytest

from studio import provenance
from studio.config import Config


@pytest.fixture
def cfg(tmp_path):
    return Config.model_validate({
        "data_root": str(tmp_path),
        "models": {
            "flux3-klein": {"repo": "o/f", "revision": "r1", "biometric_outputs": "forbidden", "files": []},
            "wan22": {"repo": "o/w", "revision": "r2", "biometric_outputs": "allowed", "files": []},
            "mystery": {"repo": "o/m", "revision": "r3", "files": []}}})


def camera_photo(cfg, name="p1.jpg"):
    folder = cfg.data_root / "cast" / "me" / "photos"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"camera " + name.encode())
    return folder / name


def made(cfg, name, models, inputs):
    path = cfg.data_root / "projects" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    provenance.write(cfg, path, kind="clip", models=models, inputs=inputs, params={"seed": 7})
    return path


def test_a_camera_original_needs_no_sidecar(cfg):
    assert provenance.problems(cfg, camera_photo(cfg)) == []


def test_a_clip_made_by_an_allowed_model_from_a_camera_photo_may_be_scored(cfg):
    crop = made(cfg, "start.png", [], [camera_photo(cfg)])
    clip = made(cfg, "clip.mp4", ["wan22"], [crop])
    assert provenance.problems(cfg, clip) == []
    record = json.loads(provenance.sidecar(clip).read_text())
    assert record["models"] == [{"set": "wan22", "revision": "r2"}] and record["params"] == {"seed": 7}
    assert (provenance.sidecar(clip).stat().st_mode & 0o777) == 0o600


def test_anything_with_a_flux_ancestor_is_refused(cfg):
    still = made(cfg, "still.png", ["flux3-klein"], [camera_photo(cfg)])
    clip = made(cfg, "clip.mp4", ["wan22"], [still])
    assert provenance.problems(cfg, clip) == ["still.png: made by flux3-klein, whose license forbids face recognition"]


def test_a_model_set_without_a_license_entry_is_refused(cfg):
    clip = made(cfg, "clip.mp4", ["mystery"], [camera_photo(cfg)])
    assert provenance.problems(cfg, clip) == ["clip.mp4: made by mystery, whose license for face recognition is unknown"]


def test_a_file_without_provenance_or_changed_since_is_refused(cfg):
    stray = cfg.data_root / "projects" / "stray.mp4"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"x")
    assert provenance.problems(cfg, stray) == ["stray.mp4: unknown origin (no provenance)"]
    clip = made(cfg, "clip.mp4", ["wan22"], [camera_photo(cfg)])
    clip.write_bytes(b"edited")
    assert provenance.problems(cfg, clip) == ["clip.mp4: changed since its provenance was written"]


def test_an_input_that_changed_or_vanished_is_refused(cfg):
    photo = camera_photo(cfg)
    crop = made(cfg, "start.png", [], [photo])
    photo.write_bytes(b"other")
    assert provenance.problems(cfg, crop) == ["start.png: input p1.jpg changed since"]
    photo.unlink()
    assert provenance.problems(cfg, crop) == ["start.png: input p1.jpg is gone"]


def test_a_file_outside_the_cast_capture_folders_is_not_a_camera_original(cfg):
    elsewhere = cfg.data_root / "cast" / "me" / "dataset" / "p1.jpg"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_bytes(b"x")
    assert provenance.problems(cfg, elsewhere) == ["p1.jpg: unknown origin (no provenance)"]


def test_model_sets_take_only_known_license_values(tmp_path):
    with pytest.raises(ValueError):
        Config.model_validate({"data_root": str(tmp_path), "models": {
            "x": {"repo": "o/x", "revision": "r", "biometric_outputs": "maybe", "files": []}}})
    cfg = Config.model_validate({"data_root": str(tmp_path), "models": {"x": {"repo": "o/x", "revision": "r",
                                                                              "files": []}}})
    assert cfg.models["x"].biometric_outputs == "unknown"

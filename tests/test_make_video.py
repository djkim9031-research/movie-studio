import pytest
from typer.testing import CliRunner

from studio.cli import app
from studio.config import load_config
from studio.gpu.preflight import Verdict
from studio.media import ffmpeg
from studio.media.smoke import smoke_mux
from studio.render import LANDSCAPE, PORTRAIT, image_size, prepare_still, target_size

CFG = load_config()
needs_ffmpeg = pytest.mark.skipif(not ffmpeg.binary(CFG).exists(), reason="run: uv run studio fetch-ffmpeg")


def test_target_size_follows_the_picture_orientation():
    assert target_size(4000, 3000) == LANDSCAPE == (1280, 704)
    assert target_size(3000, 4000) == PORTRAIT == (704, 1280)
    assert target_size(2000, 2000) == LANDSCAPE


def make_picture(path, width, height):
    ffmpeg.run(CFG, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}",
               "-frames:v", "1", str(path)).check_returncode()
    return path


@needs_ffmpeg
def test_prepare_still_crops_and_scales_to_the_target(tmp_path):
    picture = make_picture(tmp_path / "photo.jpg", 1000, 1500)
    assert image_size(CFG, picture) == (1000, 1500)
    out = prepare_still(CFG, picture, tmp_path / "still.png", PORTRAIT)
    assert image_size(CFG, out) == PORTRAIT


@needs_ffmpeg
def test_make_video_renders_a_portrait_picture_and_keeps_the_prompt_beside_the_clip(tmp_path, monkeypatch):
    picture = make_picture(tmp_path / "portrait.jpg", 900, 1600)
    out_dir = tmp_path / "comfy-io" / "output" / "studio"
    out_dir.mkdir(parents=True)
    smoke_mux(CFG, out_dir / "video_00001_.mp4")
    (tmp_path / "comfy-io" / "input").mkdir(parents=True)
    (tmp_path / "bin").symlink_to(CFG.bin_dir)
    sent = []

    class DoneClient:
        def run(self, workflow, timeout_s):
            sent.append(workflow)
            return {"outputs": {"75": {"images": [{"filename": "video_00001_.mp4", "subfolder": "studio",
                                                   "type": "output"}]}}}

    class Peak:
        peak_mib = 43008

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\n")
    monkeypatch.setattr("studio.gpu.runtime.is_running", lambda: True)
    monkeypatch.setattr("studio.gpu.runtime.container_pids", lambda: set())
    monkeypatch.setattr("studio.cli.run_preflight", lambda *a, **k: Verdict(ok=True))
    monkeypatch.setattr("studio.gpu.runtime.client", lambda cfg: DoneClient())
    monkeypatch.setattr("studio.render.VramSampler", Peak)
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "make-video", str(picture),
                                      "--prompt", "She turns toward the window and smiles.", "--seed", "11",
                                      "--name", "window"])
    assert result.exit_code == 0, result.output
    assert sent[0]["398:372"]["inputs"]["value"] == 704 and sent[0]["398:360"]["inputs"]["value"] == 1280
    clip = tmp_path / "projects" / "videos" / "window-seed11.mp4"
    assert clip.exists()
    assert "She turns toward the window and smiles." in clip.with_suffix(".txt").read_text()
    assert str(clip) in result.output


def test_make_video_refuses_a_missing_picture(tmp_path):
    cfg_file = tmp_path / "studio.yaml"
    cfg_file.write_text(f"data_root: {tmp_path}\n")
    result = CliRunner().invoke(app, ["--config", str(cfg_file), "make-video", str(tmp_path / "nope.jpg"),
                                      "--prompt", "x"])
    assert result.exit_code == 1 and "no picture at" in result.output


def test_output_paths_must_stay_inside_comfyuis_output_folder(tmp_path):
    from studio.gpu.comfy import RenderError
    from studio.render import output_path
    assert output_path(tmp_path, "studio", "a.mp4") == tmp_path / "studio" / "a.mp4"
    with pytest.raises(RenderError, match="outside"):
        output_path(tmp_path, "../..", "etc/passwd")

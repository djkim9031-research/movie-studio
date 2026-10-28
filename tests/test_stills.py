import json
import math

import pytest

from studio import stills
from studio.config import Config
from studio.refs import Reference


def make_refs(tmp_path, people=2, extras=1):
    out = []
    for i in range(1, people + 1):
        f = tmp_path / f"person{i}.png"
        f.write_bytes(b"p")
        out.append(Reference("person", f, f, f"photo{i}.jpg", 900, 300.0))
    for i in range(1, extras + 1):
        f, sq = tmp_path / f"extra{i}.png", tmp_path / f"extra{i}_square.png"
        f.write_bytes(b"e")
        sq.write_bytes(b"s")
        out.append(Reference("extra", f, sq, f"place{i}.png"))
    return out


def nodes(graph, class_type):
    return [n for n in graph.values() if n["class_type"] == class_type]


def test_graph_follows_bfls_multi_image_workflow():
    g = stills.graph(["a1.png", "a2.png", "a3.png"], ["a1.png", "a2.png", "a3_sq.png"], "Make it so.", seed=5,
                     size=(1920, 1088), prefix="scenes/x_still1")
    loads = {n["inputs"]["image"] for n in nodes(g, "LoadImage")}
    assert loads == {"a1.png", "a2.png", "a3.png", "a3_sq.png"}
    (ref,) = nodes(g, "ReferenceMultiLatent")
    assert ref["inputs"]["resolution"] == "resize" and ref["inputs"]["resolution.megapixels"] == 1.0
    assert [g[ref["inputs"][f"images.image{i}"][0]]["inputs"]["image"] for i in range(3)] == ["a1.png", "a2.png", "a3.png"]
    (batch,) = nodes(g, "BatchImagesNode")
    assert [g[batch["inputs"][f"images.image{i}"][0]]["inputs"]["image"] for i in range(3)] == [
        "a1.png", "a2.png", "a3_sq.png"]
    (upsampler,) = nodes(g, "TextGenerate")
    u = upsampler["inputs"]
    assert u["prompt"].endswith("User request: Make it so.\nOutput the caption.")
    assert (u["max_length"], u["sampling_mode"], u["sampling_mode.temperature"], u["sampling_mode.top_k"],
            u["sampling_mode.top_p"], u["thinking"], u["mtp"]) == (3584, "on", 1.0, 1, 0.95, False, "off")
    assert u["system_prompt"].startswith("You are the prompt upsampler of an image-editing model.")
    (sampler,) = nodes(g, "KSampler")
    s = sampler["inputs"]
    assert (s["seed"], s["steps"], s["cfg"], s["sampler_name"], s["scheduler"], s["denoise"]) == (
        5, 16, 1.0, "euler", "simple", 1.0)
    (shift,) = nodes(g, "ModelSamplingFlux")
    assert shift["inputs"]["max_shift"] == shift["inputs"]["base_shift"] == pytest.approx(math.log(16))
    (latent,) = nodes(g, "EmptyFlux2LatentImage")
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1920, 1088)
    (save,) = nodes(g, "SaveImage")
    assert save["inputs"]["filename_prefix"] == "scenes/x_still1"
    assert {n["inputs"].get("unet_name") for n in nodes(g, "UNETLoader")} == {"flux3-image-klein-bf16.safetensors"}
    assert {n["inputs"]["clip_name"] for n in nodes(g, "CLIPLoader")} == {
        "qwen3vl_4b_bf16.safetensors", "qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors"}


def test_graph_takes_one_to_ten_references():
    with pytest.raises(ValueError):
        stills.graph([], [], "x", seed=1, size=(1920, 1088), prefix="p")
    with pytest.raises(ValueError):
        names = [f"r{i}.png" for i in range(11)]
        stills.graph(names, names, "x", seed=1, size=(1920, 1088), prefix="p")


class FakeComfy:
    def __init__(self, data_root, cached=False):
        self.root, self.cached, self.graphs = data_root, cached, []

    def run(self, workflow, timeout_s):
        self.graphs.append(workflow)
        save_id = next((k for k, n in workflow.items() if n["class_type"] == "SaveImage"), None)
        if save_id is None:  # the upsampler alone
            return {"status": {"messages": []}, "outputs": {"34": {"text": ["A man stands on a balcony."]}}}
        out = self.root / "comfy-io" / "output" / "scenes"
        out.mkdir(parents=True, exist_ok=True)
        (out / "x_still1_00001_.png").write_bytes(b"png")
        status = {"messages": [["execution_cached", {"nodes": [save_id] if self.cached else []}]]}
        return {"status": status, "outputs": {
            "34": {"text": ["Create a photograph of the person in images one and two."]},
            save_id: {"images": [{"filename": "x_still1_00001_.png", "subfolder": "scenes", "type": "output"}]}}}


def test_render_still_copies_in_runs_copies_out_and_leaves_nothing_in_comfy_io(tmp_path, monkeypatch):
    cfg = Config.model_validate({"data_root": str(tmp_path)})
    fake = FakeComfy(tmp_path)
    monkeypatch.setattr("studio.gpu.runtime.client", lambda c: fake)
    references = make_refs(tmp_path)
    dest = tmp_path / "scene" / "stills" / "still1.png"
    dest.parent.mkdir(parents=True)
    result = stills.render_still(cfg, references, "Make it so.", seed=9, size=(1920, 1088), name="x_still1",
                                 dest=dest)
    assert dest.read_bytes() == b"png"
    assert result.caption == "Create a photograph of the person in images one and two."
    loads = sorted(n["inputs"]["image"] for n in nodes(fake.graphs[0], "LoadImage"))
    assert loads == ["x_still1_ref1.png", "x_still1_ref2.png", "x_still1_ref3.png", "x_still1_ref3_square.png"]
    assert list((tmp_path / "comfy-io" / "input").iterdir()) == []
    assert list((tmp_path / "comfy-io" / "output" / "scenes").iterdir()) == []
    assert json.loads(json.dumps(result.graph)) == fake.graphs[0]


def test_render_still_refuses_an_image_comfyui_took_from_an_earlier_run(tmp_path, monkeypatch):
    cfg = Config.model_validate({"data_root": str(tmp_path)})
    monkeypatch.setattr("studio.gpu.runtime.client", lambda c: FakeComfy(tmp_path, cached=True))
    with pytest.raises(stills.RenderError, match="earlier run"):
        stills.render_still(cfg, make_refs(tmp_path), "x", seed=9, size=(1920, 1088), name="x_still1",
                            dest=tmp_path / "out.png")


def test_a_graph_with_a_caption_draws_from_it_without_the_upsampler():
    g = stills.graph(["a.png", "b.png"], ["a_sq.png", "b_sq.png"], "Make it so.", seed=3, size=(1920, 1088),
                     prefix="p", caption="A man stands on a balcony.")
    assert not nodes(g, "TextGenerate") and not nodes(g, "PreviewAny") and not nodes(g, "BatchImagesNode")
    assert g["44"]["inputs"]["text"] == "A man stands on a balcony."
    assert sorted(n["inputs"]["image"] for n in nodes(g, "LoadImage")) == ["a.png", "b.png"]
    assert len([k for k in g["46"]["inputs"] if k.startswith("images.")]) == 2


def test_upsample_runs_only_the_prompt_upsampler_and_leaves_nothing_in_comfy_io(tmp_path, monkeypatch):
    cfg = Config.model_validate({"data_root": str(tmp_path)})
    fake = FakeComfy(tmp_path)
    monkeypatch.setattr("studio.gpu.runtime.client", lambda c: fake)
    caption = stills.upsample(cfg, make_refs(tmp_path), "Make it so.", name="x_still1")
    assert caption == "A man stands on a balcony."
    g = fake.graphs[0]
    assert nodes(g, "TextGenerate") and not nodes(g, "SaveImage") and not nodes(g, "KSampler")
    assert "Make it so." in nodes(g, "TextGenerate")[0]["inputs"]["prompt"]
    assert sorted(n["inputs"]["image"] for n in nodes(g, "LoadImage")) == \
        ["x_still1_up1.png", "x_still1_up2.png", "x_still1_up3.png"]
    assert list((tmp_path / "comfy-io" / "input").iterdir()) == []


def test_render_still_with_a_caption_draws_and_reports_that_caption(tmp_path, monkeypatch):
    cfg = Config.model_validate({"data_root": str(tmp_path)})
    fake = FakeComfy(tmp_path)
    monkeypatch.setattr("studio.gpu.runtime.client", lambda c: fake)
    dest = tmp_path / "still1.png"
    result = stills.render_still(cfg, make_refs(tmp_path), "Make it so.", seed=9, size=(1920, 1088),
                                 name="x_still1", dest=dest, caption="A man on a balcony, patched.")
    assert result.caption == "A man on a balcony, patched." and dest.read_bytes() == b"png"
    assert fake.graphs[0]["44"]["inputs"]["text"] == "A man on a balcony, patched."
    assert sorted(n["inputs"]["image"] for n in nodes(fake.graphs[0], "LoadImage")) == \
        ["x_still1_ref1.png", "x_still1_ref2.png", "x_still1_ref3.png"]


def sky(size=(1920, 1088)):
    """A smooth sky over a textured shore and a thin parapet line, as float RGB."""
    import numpy as np

    w, h = size
    rows = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = np.broadcast_to(np.array([120, 150, 200], np.float32) * (1 - rows) + np.array([210, 190, 190]) * rows,
                          (h, w, 3)).copy()
    img[700:760] = 60 + 30 * np.random.default_rng(1).random((60, w, 1))
    img[1000:1003] = 230
    return img


def blob(img, cx, cy, r=30, lift=120.0):
    import numpy as np

    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    return img + lift * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r**2))[..., None]


def test_only_what_the_edit_added_near_the_box_is_kept():
    import numpy as np

    base = sky()
    drift = np.array([5.0, 1.0, -2.0], np.float32)  # the redrawn frame's colour shift
    edit = base + drift
    edit = blob(edit, 0.73 * 1920, 0.2 * 1088)  # beside the box, still near it
    edit = blob(edit, 200, 200)  # far from the box
    edit[1000:1003], edit[1002:1005] = base[1003:1006] + drift, 230  # the parapet moved by 2 px
    out, found = stills.blend_event(base, edit, [0.60, 0.10, 0.70, 0.30])
    assert found > 1000
    assert out[int(0.2 * 1088), int(0.73 * 1920)] == pytest.approx(base[int(0.2 * 1088), int(0.73 * 1920)] + 120, abs=6)
    assert np.array_equal(out[150:250, 150:250], base[150:250, 150:250])  # the far change stays out
    assert np.array_equal(out[990:1010], base[990:1010])
    assert np.array_equal(out[:, :800], base[:, :800])


def test_an_edit_that_adds_nothing_near_the_box_finds_no_event():
    import numpy as np

    base = sky()
    out, found = stills.blend_event(base, base + np.array([9.0, 0.0, -8.0], np.float32), [0.60, 0.10, 0.70, 0.30])
    assert found == 0 and np.array_equal(out, base)


def test_paint_event_edits_the_whole_frame_from_the_base_and_keeps_the_base_elsewhere(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    cfg = Config.model_validate({"data_root": str(tmp_path)})
    base_pixels = sky()

    class Editor(FakeComfy):
        def run(self, workflow, timeout_s):
            entry = super().run(workflow, timeout_s)
            edited = blob(base_pixels + 3.0, 0.65 * 1920, 0.2 * 1088, lift=60.0)
            Image.fromarray(edited.round().clip(0, 255).astype(np.uint8)).save(
                self.root / "comfy-io" / "output" / "scenes" / "x_still1_00001_.png")
            return entry

    fake = Editor(tmp_path)
    monkeypatch.setattr("studio.gpu.runtime.client", lambda c: fake)
    base = tmp_path / "base.png"
    Image.fromarray(base_pixels.round().astype(np.uint8)).save(base)
    dest = tmp_path / "painted.png"
    told = "Edit the first image: add a small flash on the horizon. Keep everything else exactly the same."
    result = stills.paint_event(cfg, base, [0.60, 0.10, 0.70, 0.30], make_refs(tmp_path, people=0, extras=1), told,
                                seed=4, name="x_still1", dest=dest)
    (g,) = fake.graphs
    assert not nodes(g, "SetLatentNoiseMask") and not nodes(g, "ImageCompositeMasked") and not nodes(g, "TextGenerate")
    assert g["44"]["inputs"]["text"] == told and nodes(g, "KSampler")[0]["inputs"]["seed"] == 4
    (latent,) = nodes(g, "EmptyFlux2LatentImage")
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1920, 1088)
    refs = [g[v[0]]["inputs"]["image"] for k, v in g["46"]["inputs"].items() if k.startswith("images.")]
    assert refs == ["x_still1_ref1.png", "x_still1_ref2.png"]  # the base first
    painted = np.asarray(Image.open(dest)).astype(np.float32)
    kept = np.asarray(Image.open(base)).astype(np.float32)
    assert np.array_equal(painted[:, :800], kept[:, :800])
    assert painted[int(0.2 * 1088), int(0.65 * 1920)].mean() == pytest.approx(
        kept[int(0.2 * 1088), int(0.65 * 1920)].mean() + 60, abs=4)
    assert result.event_px > 1000 and result.edit == tmp_path / "painted-edit.png" and result.edit.exists()
    assert list((tmp_path / "comfy-io" / "input").iterdir()) == []
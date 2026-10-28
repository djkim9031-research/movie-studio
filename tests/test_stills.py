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


def test_paint_graph_draws_only_inside_the_mask_and_pastes_back_through_it():
    g = stills.paint_graph("base.png", "mask.png", ["blast.png"], "A small flash on the horizon.", seed=4,
                           prefix="p")
    by = {k: n for k, n in g.items()}
    load = {n["inputs"]["image"]: k for k, n in g.items() if n["class_type"] == "LoadImage"}
    assert set(load) == {"base.png", "mask.png", "blast.png"}
    assert not nodes(g, "TextGenerate") and g["44"]["inputs"]["text"] == "A small flash on the horizon."
    mask = next(k for k, n in g.items() if n["class_type"] == "ImageToMask")
    assert by[mask]["inputs"]["image"] == [load["mask.png"], 0]
    encode = next(k for k, n in g.items() if n["class_type"] == "VAEEncode")
    assert by[encode]["inputs"]["pixels"] == [load["base.png"], 0]
    noise = next(k for k, n in g.items() if n["class_type"] == "SetLatentNoiseMask")
    assert by[noise]["inputs"] == {"samples": [encode, 0], "mask": [mask, 0]}
    assert nodes(g, "KSampler")[0]["inputs"]["latent_image"] == [noise, 0]
    refs = [v for k, v in by["46"]["inputs"].items() if k.startswith("images.")]
    assert refs == [[load["base.png"], 0], [load["blast.png"], 0]]  # the base first, for its light and color
    paste = nodes(g, "ImageCompositeMasked")[0]["inputs"]
    assert paste["destination"] == [load["base.png"], 0] and paste["mask"] == [mask, 0]
    assert nodes(g, "SaveImage")[0]["inputs"]["images"] == [next(k for k, n in g.items()
                                                                 if n["class_type"] == "ImageCompositeMasked"), 0]


def test_paint_event_masks_the_box_with_a_soft_edge_and_leaves_nothing_in_comfy_io(tmp_path, monkeypatch):
    from PIL import Image

    cfg = Config.model_validate({"data_root": str(tmp_path)})
    seen = {}

    class Painter(FakeComfy):
        def run(self, workflow, timeout_s):
            mask_name = next(n["inputs"]["image"] for n in workflow.values()
                             if n["class_type"] == "LoadImage" and n["inputs"]["image"].endswith("_mask.png"))
            with Image.open(tmp_path / "comfy-io" / "input" / mask_name) as m:
                seen["mode"], seen["mask"] = m.mode, m.convert("L").copy()
            return super().run(workflow, timeout_s)

    monkeypatch.setattr("studio.gpu.runtime.client", lambda c: Painter(tmp_path))
    base = tmp_path / "base.png"
    Image.new("RGB", (1920, 1088), (20, 40, 60)).save(base)
    dest = tmp_path / "painted.png"
    result = stills.paint_event(cfg, base, [0.70, 0.30, 0.74, 0.36], make_refs(tmp_path, people=0, extras=1),
                                "A small flash on the horizon.", seed=4, name="x_still1", dest=dest)
    mask, mode = seen["mask"], seen["mode"]
    assert mask.size == (1920, 1088) and mode == "RGB"  # LoadImage reads RGB exactly
    x0, y0, x1, y1 = round(0.70 * 1920), round(0.30 * 1088), round(0.74 * 1920), round(0.36 * 1088)
    assert min(mask.getpixel((x, y)) for x in range(x0, x1, 4) for y in range(y0, y1, 4)) >= 240  # the whole box
    assert mask.getpixel((100, 100)) == 0 and mask.getpixel(((x0 + x1) // 2, y1 + 60)) == 0
    assert 0 < mask.getpixel((x0 - 6, (y0 + y1) // 2)) < 255  # a soft edge just outside
    assert dest.read_bytes() == b"png" and result.caption == "A small flash on the horizon."
    assert list((tmp_path / "comfy-io" / "input").iterdir()) == []


def test_a_large_event_box_keeps_its_soft_edge_close(tmp_path):
    from PIL import Image

    out = stills.event_mask((1920, 1088), [0.2, 0.2, 0.5, 0.7], tmp_path / "mask.png")
    with Image.open(out) as m:
        mask = m.convert("L")
    x1 = round(0.5 * 1920)
    assert mask.getpixel((x1 - 2, 500)) >= 240 and mask.getpixel((x1 + 48, 500)) == 0

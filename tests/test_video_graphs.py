import pytest

from studio import video_graphs as vg
from studio.config import Config


def by_class(graph, class_type):
    return [(k, n["inputs"]) for k, n in graph.items() if n["class_type"] == class_type]


def test_ltx_single_stage_pins_the_first_frame_at_full_size():
    g = vg.ltx_single(prompt="p", first="a.png", size=(1280, 704), seconds=5, seed=3,
                      unet="ltx-2.5-22b-distilled-transformer-bf16.safetensors", prefix="bench/x")
    (_, latent), = by_class(g, "EmptyLTXVLatentVideo")
    assert (latent["width"], latent["height"], latent["length"]) == (1280, 704, 121)
    (_, guide), = by_class(g, "LTXVAddGuide")
    assert (guide["frame_idx"], guide["strength"]) == (0, 1.0)
    assert g[guide["image"][0]]["class_type"] == "LTXVPreprocess"
    (_, sigmas), = by_class(g, "ManualSigmas")
    assert len(sigmas["sigmas"].split(",")) == 9
    (_, unet), = by_class(g, "UNETLoader")
    assert unet["unet_name"].endswith("bf16.safetensors")
    (_, noise), = by_class(g, "RandomNoise")
    assert noise["noise_seed"] == 3
    assert len(by_class(g, "SaveVideo")) == 1 and not by_class(g, "LTXVLatentUpsampler")


def test_ltx_single_stage_with_a_last_frame_adds_a_second_guide_at_the_end():
    g = vg.ltx_single(prompt="p", first="a.png", last="b.png", size=(1280, 704), seconds=5, seed=3,
                      unet="u.safetensors", prefix="x")
    guides = sorted((i["frame_idx"], g[g[i["image"][0]]["inputs"]["image"][0]]["inputs"]["image"])
                    for _, i in by_class(g, "LTXVAddGuide"))
    assert guides == [(-1, "b.png"), (0, "a.png")]
    (_, crop), = by_class(g, "LTXVCropGuides")
    last_guide = [k for k, i in by_class(g, "LTXVAddGuide") if i["frame_idx"] == -1][0]
    assert crop["positive"] == [last_guide, 0]


def test_wan_four_step_uses_both_experts_with_the_lora_and_splits_at_step_two():
    g = vg.wan_i2v(prompt="p", start="a.png", size=(1280, 704), seconds=5, seed=9, steps=4, cfg=1.0, split=2,
                   loras=vg.WAN_LORAS_1022, prefix="x")
    (_, i2v), = by_class(g, "WanImageToVideo")
    assert (i2v["width"], i2v["height"], i2v["length"]) == (1280, 704, 81)
    high, low = sorted(by_class(g, "KSamplerAdvanced"), key=lambda kv: kv[1]["start_at_step"])
    assert (high[1]["add_noise"], high[1]["noise_seed"], high[1]["end_at_step"]) == ("enable", 9, 2)
    assert high[1]["return_with_leftover_noise"] == "enable"
    assert (low[1]["add_noise"], low[1]["start_at_step"], low[1]["end_at_step"]) == ("disable", 2, 4)
    assert low[1]["latent_image"] == [high[0], 0]
    assert sorted(i["lora_name"] for _, i in by_class(g, "LoraLoaderModelOnly")) == sorted(vg.WAN_LORAS_1022)
    (_, video), = by_class(g, "CreateVideo")
    assert video["fps"] == 16.0


def test_wan_without_loras_and_with_an_end_frame_uses_first_last_frame():
    g = vg.wan_i2v(prompt="p", start="a.png", end="b.png", size=(1280, 704), seconds=5, seed=9, steps=20, cfg=3.5,
                   split=10, loras=None, prefix="x")
    assert not by_class(g, "LoraLoaderModelOnly") and not by_class(g, "WanImageToVideo")
    (_, flf), = by_class(g, "WanFirstLastFrameToVideo")
    assert g[flf["end_image"][0]]["inputs"]["image"] == "b.png"
    (_, neg), = [(k, n["inputs"]) for k, n in g.items() if n["class_type"] == "CLIPTextEncode"
                 and n["inputs"]["text"] == vg.WAN_NEGATIVE]
    assert neg


class FakeComfy:
    def __init__(self, root):
        self.root, self.graphs = root, []

    def run(self, workflow, timeout_s):
        self.graphs.append(workflow)
        out = self.root / "comfy-io" / "output" / "bench"
        out.mkdir(parents=True, exist_ok=True)
        (out / "x_00001_.mp4").write_bytes(b"mp4")
        save = next(k for k, n in workflow.items() if n["class_type"] == "SaveVideo")
        return {"outputs": {save: {"images": [{"filename": "x_00001_.mp4", "subfolder": "bench", "type": "output"}]}}}


def test_render_graph_copies_images_in_and_the_clip_out_and_cleans_up(tmp_path, monkeypatch):
    cfg = Config.model_validate({"data_root": str(tmp_path)})
    fake = FakeComfy(tmp_path)
    monkeypatch.setattr("studio.gpu.runtime.client", lambda c: fake)
    monkeypatch.setattr("studio.video_graphs.VramSampler", vg.NoSampler)
    start = tmp_path / "start.png"
    start.write_bytes(b"png")
    g = vg.ltx_single(prompt="p", first="START", size=(1280, 704), seconds=5, seed=3, unet="u", prefix="bench/x")
    dest = tmp_path / "clip.mp4"
    result = vg.render_graph(cfg, g, {"START": start}, name="x", dest=dest)
    assert dest.read_bytes() == b"mp4" and result.elapsed_s >= 0
    loads = [n["inputs"]["image"] for n in fake.graphs[0].values() if n["class_type"] == "LoadImage"]
    assert loads == ["x_START.png"]
    assert list((tmp_path / "comfy-io" / "input").iterdir()) == []
    assert list((tmp_path / "comfy-io" / "output" / "bench").iterdir()) == []


def test_render_graph_refuses_a_placeholder_without_an_image(tmp_path):
    cfg = Config.model_validate({"data_root": str(tmp_path)})
    g = vg.ltx_single(prompt="p", first="START", size=(1280, 704), seconds=5, seed=3, unet="u", prefix="x")
    with pytest.raises(ValueError, match="START"):
        vg.render_graph(cfg, g, {}, name="x", dest=tmp_path / "c.mp4")

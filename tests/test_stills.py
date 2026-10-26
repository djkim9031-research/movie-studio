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
        out = self.root / "comfy-io" / "output" / "scenes"
        out.mkdir(parents=True, exist_ok=True)
        (out / "x_still1_00001_.png").write_bytes(b"png")
        save_id = next(k for k, n in workflow.items() if n["class_type"] == "SaveImage")
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

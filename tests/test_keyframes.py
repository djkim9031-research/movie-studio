import pytest

from studio import keyframes as kf


def by_class(graph, class_type):
    return [(k, n["inputs"]) for k, n in graph.items() if n["class_type"] == class_type]


def loaded(graph, link):
    node = graph[link[0]]
    while node["class_type"] != "LoadImage":
        node = graph[node["inputs"]["image"][0]]
    return node["inputs"]["image"]


def test_qwen_edit_takes_up_to_three_references_into_both_text_encoders():
    g = kf.qwen_edit(prompt="Create a photograph of the person in image 1.", refs=["R1", "R2", "R3"],
                     size=(1664, 928), seed=4, prefix="keyframes/x")
    encoders = by_class(g, "TextEncodeQwenImageEditPlus")
    assert len(encoders) == 2 and {e["prompt"] for _, e in encoders} == {"Create a photograph of the person in image 1.", ""}
    for _, e in encoders:
        assert [loaded(g, e[f"image{i}"]) for i in (1, 2, 3)] == ["R1", "R2", "R3"]
    (_, sampler), = by_class(g, "KSampler")
    assert (sampler["steps"], sampler["cfg"], sampler["seed"]) == (40, 4.0, 4)
    (_, latent), = by_class(g, "EmptySD3LatentImage")
    assert (latent["width"], latent["height"]) == (1664, 928)
    (_, shift), = by_class(g, "ModelSamplingAuraFlow")
    assert shift["shift"] == pytest.approx(3.1)
    (_, unet), = by_class(g, "UNETLoader")
    assert unet["unet_name"] == "qwen_image_edit_2511_bf16.safetensors"
    with pytest.raises(ValueError):
        kf.qwen_edit(prompt="p", refs=["a", "b", "c", "d"], size=(1664, 928), seed=1, prefix="x")


def test_flux2_klein_4b_chains_one_reference_latent_per_image_on_both_sides():
    g = kf.flux2_klein(prompt="p", refs=["R1", "R2", "R3", "R4"], size=(1920, 1088), seed=5, prefix="x")
    refs = by_class(g, "ReferenceLatent")
    assert len(refs) == 8
    (_, guider), = by_class(g, "CFGGuider")
    assert guider["cfg"] == 1.0
    (_, sched), = by_class(g, "Flux2Scheduler")
    assert (sched["steps"], sched["width"], sched["height"]) == (4, 1920, 1088)
    (_, unet), = by_class(g, "UNETLoader")
    assert unet["unet_name"] == "flux-2-klein-4b.safetensors"
    with pytest.raises(ValueError):
        kf.flux2_klein(prompt="p", refs=["a"] * 5, size=(1920, 1088), seed=1, prefix="x")

"""ComfyUI API-format workflows: fill named slots ("node_id.input_name") and model frame-count rules."""

import copy

ENHANCER = "TextGenerateLTX2Prompt"
SWITCH = "ComfySwitchNode"
PREVIEW = "PreviewAny"


def _links(node: dict) -> dict:
    """Inputs that come from another node: name -> [node_id, output_index]."""
    return {name: value for name, value in node["inputs"].items()
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)}


def _consumers(workflow: dict, node_id: str) -> list[str]:
    return [cid for cid, node in workflow.items() if any(v[0] == node_id for v in _links(node).values())]


def strip_prompt_enhancer(workflow: dict) -> dict:
    """A copy without LTX's prompt enhancer, whose model ComfyUI would otherwise demand even when it is switched off.

    Each switch between the raw prompt (on_false) and the enhanced one (on_true) is replaced by the raw prompt; then
    previews of removed nodes, and nodes that fed only removed nodes, are dropped too.
    """
    wf = copy.deepcopy(workflow)
    enhancers = {nid for nid, node in wf.items() if node["class_type"] == ENHANCER}
    switches = {nid for nid, node in wf.items()
                if node["class_type"] == SWITCH and _links(node).get("on_true", [None])[0] in enhancers}
    for sid in switches:
        raw = wf[sid]["inputs"]["on_false"]
        for cid in _consumers(wf, sid):
            if wf[cid]["class_type"] != PREVIEW:
                for name, value in _links(wf[cid]).items():
                    if value[0] == sid:
                        wf[cid]["inputs"][name] = list(raw)
    removed = enhancers | switches
    while True:
        orphans = set()
        for nid, node in wf.items():
            if nid in removed:
                continue
            consumers = _consumers(wf, nid)
            if (node["class_type"] == PREVIEW and any(v[0] in removed for v in _links(node).values())) \
                    or (consumers and all(c in removed for c in consumers)):
                orphans.add(nid)
        if not orphans:
            break
        removed |= orphans
    return {nid: node for nid, node in wf.items() if nid not in removed}


def ltx_frames(seconds: float, fps: int = 24) -> int:
    """LTX takes 8n+1 frames: the count nearest `seconds` at `fps`."""
    return 8 * round(seconds * fps / 8) + 1


def fill(template: dict, slots: dict[str, str], values: dict) -> dict:
    """A copy of `template` with each value written to its slot; unknown names or targets raise KeyError."""
    workflow = copy.deepcopy(template)
    for name, value in values.items():
        if name not in slots:
            raise KeyError(f"no slot named {name}")
        node_id, _, input_name = slots[name].partition(".")
        if input_name not in workflow.get(node_id, {}).get("inputs", {}):
            raise KeyError(f"slot {name} points at {slots[name]}, which the workflow lacks")
        workflow[node_id]["inputs"][input_name] = value
    return workflow

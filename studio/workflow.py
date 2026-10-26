"""ComfyUI API-format workflows: fill named slots ("node_id.input_name") and model frame-count rules."""

import copy


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

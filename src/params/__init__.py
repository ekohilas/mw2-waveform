"""Scene parameters: schema + JSON load/save."""
import json
import os
import re

from .schema import (Ambient, Echo, GainGrid, Glow, Layer, Look, Packet, Scene, Smudge,
                     Strand, StrandJitter, StrandSet, keyed, scene_from_dict, scene_to_dict)

DEFAULT_SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.json")


def load_scene(path: str = None) -> Scene:
    with open(path or DEFAULT_SCENE) as f:
        return scene_from_dict(json.load(f))


def _round(o, nd=3):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, list):
        return [_round(e, nd) for e in o]
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    return o


def save_scene(scene: Scene, path: str = None, ndigits: int = 3) -> None:
    """Write compact JSON (floats rounded, one packet per line)."""
    d = _round(scene_to_dict(scene), ndigits)
    packets = d.pop("packets")
    smudges = d.pop("smudges")
    events = d.pop("events", [])
    text = json.dumps(d, indent=1)
    # collapse innermost lists (numbers / short lists) onto one line
    text = re.sub(r"\[[^\[\]{}]*\]", lambda m: json.dumps(json.loads(m.group(0)),
                                                       separators=(",", ":")), text)
    text = re.sub(r"\[\s*(\[[^{}]*?\])\s*\]",
                  lambda m: json.dumps(json.loads(m.group(0)), separators=(",", ":")), text)
    def block(name, items):
        body = ",\n".join("  " + json.dumps(p, separators=(",", ":")) for p in items)
        return f',\n "{name}": [\n{body}\n ]' if items else f',\n "{name}": []'
    text = text[:-2] + block("packets", packets) + block("smudges", smudges) + block("events", events) + "\n}\n"
    with open(path or DEFAULT_SCENE, "w") as f:
        f.write(text)


__all__ = ["Ambient", "Echo", "GainGrid", "Glow", "Layer", "Look", "Packet", "Scene", "Smudge",
           "Strand", "StrandJitter", "StrandSet", "keyed", "load_scene", "save_scene",
           "scene_from_dict", "scene_to_dict", "DEFAULT_SCENE"]

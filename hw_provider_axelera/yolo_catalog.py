# Copyright 2026 SustainML Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""YOLO models offered for CNNs on the Axelera board, and their hardware resources.

The models, their goals and their figures live in yolo_models.json. Each model has the
accuracy and FPS Axelera publishes (reference) and the values measured on our board or computed
from its ONNX file (measured, how in its "measurement"). Measured values are used when present,
placeholders where not yet: the latency from the full-load FPS, and, while the board's power is
not measured (it has no power sensor), the datasheet typical power of axelera_devices.yaml. The
result says where each value comes from in its "source".
"""

import json
import os

import yaml

HERE = os.path.dirname(__file__)
CATALOG_FILE = os.path.join(HERE, "yolo_models.json")
DEVICES_FILE = os.path.join(HERE, "axelera_devices.yaml")

AXELERA_HW = "Axelera Edge 130p"


def get_devices() -> dict:
    with open(DEVICES_FILE, "r") as f:
        return yaml.safe_load(f)


def load_catalog() -> dict:
    with open(CATALOG_FILE, "r") as f:
        return json.load(f)


def is_axelera(hardware) -> bool:
    return (hardware or "").strip().lower() == AXELERA_HW.lower()


def list_goals() -> list:
    return list(load_catalog()["goals"].keys())


def goal_descriptions() -> dict:
    return {name: g["description"] for name, g in load_catalog()["goals"].items()}


def list_models(goal: str = "") -> list:
    """Models of a goal (all of them when the goal is empty), in catalog order."""
    models = load_catalog()["models"]
    return [m for m in models if not goal or m["goal"] == goal]


def get_model(name: str):
    for m in load_catalog()["models"]:
        if m["name"] == name:
            return m
    return None


def resources(model: dict) -> dict:
    """Latency (s per inference), power (W) and memory (MB) of a model on the board, defined as
    for every hardware in SustainML.

    The latency is that of one input processed alone, so that power x latency is the energy of
    one inference, which is how WP3 turns them into energy. The memory is the model's weights plus
    the working memory of one input (model_memory.py). Besides, full_load_fps is the board's
    throughput: inferences per second with all its AIPU cores busy, each one on its own input.
    A value not measured yet is a placeholder (the catalog's "measurement" says which), and
    "source" says so.
    """
    catalog = load_catalog()
    measured = model.get("measured") or {}

    full_load_fps = measured.get("full_load_fps")
    if full_load_fps:
        full_load_source = "measured"
    else:
        full_load_fps = float(model["reference"]["pcie_fps"])
        full_load_source = f"not measured: reference FPS ({model['reference']['source']})"

    latency = measured.get("latency_s")
    if latency:
        latency_source = "measured"
    else:
        # How much longer one input alone takes than one inference at full load, in the models
        # measured both ways
        ratios = [m["measured"]["latency_s"] * m["measured"]["full_load_fps"] for m in catalog["models"]
                  if m["measured"].get("latency_s") and m["measured"].get("full_load_fps")]
        ratio = sum(ratios) / len(ratios) if ratios else float(catalog["measurement"].get("aipu_cores", 4))
        latency = ratio / float(full_load_fps)
        latency_source = (f"placeholder, not measured: full-load time per inference x {ratio:.2f}, "
                          f"the ratio of the {len(ratios)} model(s) measured both ways")

    power = measured.get("power_w")
    if power:
        power_source = "measured"
    else:
        spec = get_devices()[AXELERA_HW]
        power = (float(spec["typical_power_w_min"]) + float(spec["typical_power_w_max"])) / 2.0
        power_source = (f"placeholder, not measured (no power sensor on the board): datasheet typical "
                        f"{spec['typical_power_w_min']}-{spec['typical_power_w_max']} W, midpoint")

    return {
        "latency_s": float(latency),
        "full_load_fps": float(full_load_fps),
        "power_w": float(power),
        "energy_per_inference_j": float(power) * float(latency),
        "weights_mb": measured.get("weights_mb"),
        "working_mb": measured.get("working_mb"),
        "memory_mb": measured.get("memory_mb"),
        "source": {"latency": latency_source, "full_load_fps": full_load_source, "power": power_source,
                   "memory": "computed from the ONNX file" if measured.get("memory_mb") is not None
                   else "unknown"},
    }


PRIORITIES = ("accuracy", "speed", "energy", "balanced")
DEFAULT_PREFERENCES = {"priority": "balanced", "min_fps": None, "max_latency_ms": None}


def normalize_preferences(prefs) -> dict:
    """The requirements read from the problem description, made safe to use: a known priority
    and positive numbers (or None) for the speed requirements."""
    out = dict(DEFAULT_PREFERENCES)
    if not isinstance(prefs, dict):
        return out
    if str(prefs.get("priority", "")).strip().lower() in PRIORITIES:
        out["priority"] = str(prefs["priority"]).strip().lower()
    for key in ("min_fps", "max_latency_ms"):
        try:
            value = float(prefs.get(key))
            out[key] = value if value > 0 else None
        except (TypeError, ValueError):
            pass
    if prefs.get("reason"):
        out["reason"] = str(prefs["reason"])
    return out


def rank_models(goal: str, exclude=(), max_memory_mb: float = 0.0, preferences=None) -> list:
    """Models of the goal, best first, for the requirements read from the problem description.

    Models already tried (exclude) and those over the memory limit are left out (a model whose
    memory is not known yet is kept). Those too slow for the description's min_fps or
    max_latency_ms are left out too, unless none is fast enough: then all are kept, fastest
    first, as the closest ones. When some are fast enough, a speed priority becomes balanced,
    as they all meet the speed asked. The rest are ordered by the description's priority:
      accuracy: most accurate after the int8 quantization loss, lower energy breaking ties
      speed:    lowest latency per inference
      energy:   lowest energy per inference
      balanced: best accuracy and energy together, both scaled 0-1 among the candidates
    """
    prefs = normalize_preferences(preferences)
    candidates = []
    for m in list_models(goal):
        if m["name"] in exclude:
            continue
        r = resources(m)
        if max_memory_mb and r["memory_mb"] is not None and r["memory_mb"] > max_memory_mb:
            continue
        candidates.append((m, r))
    if not candidates:
        return []

    # A frame rate is a throughput (all cores together); a time per frame is what one frame
    # takes alone, its latency
    def fast_enough(r):
        if prefs["min_fps"] and r["full_load_fps"] < prefs["min_fps"]:
            return False
        if prefs["max_latency_ms"] and r["latency_s"] > prefs["max_latency_ms"] / 1000.0:
            return False
        return True

    if prefs["min_fps"] or prefs["max_latency_ms"]:
        meeting = [(m, r) for m, r in candidates if fast_enough(r)]
        if meeting:
            candidates = meeting
            # The stated speed is what the problem needs, and every model left meets it: being
            # faster still gains nothing, so the speed priority gives way to accuracy and energy
            if prefs["priority"] == "speed":
                prefs["priority"] = "balanced"
        else:
            print(f"[Axelera] No {goal} model meets min_fps={prefs['min_fps']} / "
                  f"max_latency_ms={prefs['max_latency_ms']}: keeping the fastest first")
            candidates.sort(key=lambda mr: mr[1]["latency_s"])
            return [m for m, _ in candidates]

    def accuracy(mr):
        return mr[0]["accuracy"]["int8"]

    def energy(mr):
        return mr[1]["energy_per_inference_j"]

    if prefs["priority"] == "accuracy":
        candidates.sort(key=lambda mr: (-accuracy(mr), energy(mr)))
    elif prefs["priority"] == "speed":
        candidates.sort(key=lambda mr: (mr[1]["latency_s"], -accuracy(mr)))
    elif prefs["priority"] == "energy":
        candidates.sort(key=lambda mr: (energy(mr), -accuracy(mr)))
    else:
        acc = [accuracy(mr) for mr in candidates]
        en = [energy(mr) for mr in candidates]

        def scaled(value, values):
            span = max(values) - min(values)
            return (value - min(values)) / span if span > 0 else 0.0

        candidates.sort(key=lambda mr: (-(scaled(accuracy(mr), acc) - scaled(energy(mr), en)), -accuracy(mr)))
    return [m for m, _ in candidates]

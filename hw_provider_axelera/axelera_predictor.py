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
"""Datasheet-only latency/power ESTIMATE for Axelera Metis AIPU hardware.

No physical Axelera board is used here. Numbers come from the published
datasheets in axelera_devices.yaml (peak TOPS, TDP) plus a fixed, conservative
utilization assumption, because real-world accelerator utilization is always
far below peak TOPS. Replace ASSUMED_UTILIZATION and the power figure with
real measurements once hardware is available.
"""

import os
import yaml

HERE = os.path.dirname(__file__)
DEVICES_FILE = os.path.join(HERE, "axelera_devices.yaml")

# Rough placeholder: fraction of peak INT8 TOPS assumed achievable in
# practice for a typical inference workload. Not measured, not vendor data.
ASSUMED_UTILIZATION = 0.20


def get_devices() -> dict:
    with open(DEVICES_FILE, "r") as f:
        return yaml.safe_load(f)


def estimate_flops_per_forward(model) -> float:
    """2 * parameter count: standard order-of-magnitude estimate for a
    single inference forward pass (one multiply + one add per parameter)."""
    num_params = sum(p.numel() for p in model.parameters())
    return 2.0 * num_params


def _device_power_w(spec: dict) -> float:
    if "tdp_w" in spec:
        return float(spec["tdp_w"])
    lo = spec.get("typical_power_w_min")
    hi = spec.get("typical_power_w_max")
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2.0
    return 0.0


def predict_latency_power(model, device_name: str) -> dict:
    devices = get_devices()
    if device_name not in devices:
        raise ValueError(
            f"Unknown Axelera device '{device_name}'. Known devices: {list(devices.keys())}"
        )

    spec = devices[device_name]
    peak_ops_per_s = spec["peak_tops_int8"] * 1e12 * ASSUMED_UTILIZATION

    flops = estimate_flops_per_forward(model)
    latency_s = flops / peak_ops_per_s if peak_ops_per_s > 0 else 0.0
    latency_h = latency_s / 3600.0  # WP3 expects hours, like the other providers

    power_w = _device_power_w(spec)

    return {
        "device": device_name,
        "latency_h": latency_h,
        "power_w": power_w,
        "provenance": {
            "source": "datasheet_estimate",
            "note": "No physical Axelera board was used. Rough estimate from "
                    "published TOPS/TDP figures and a fixed utilization "
                    "assumption, pending real hardware measurement.",
            "assumed_utilization": ASSUMED_UTILIZATION,
            "peak_tops_int8": spec["peak_tops_int8"],
            "power_w_used": power_w,
        },
    }

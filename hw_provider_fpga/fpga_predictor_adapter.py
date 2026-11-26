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
"""SustainML FPGA Predictor Adapter Implementation."""

import hashlib
import json
import os
import threading

from .vendor.sustain_ml_predictor.predictor import (  # Vendored DFKI code
    DEFAULT_FPS, DEFAULT_IMPLEMENTATION, FPS_OPTIONS, IMPLEMENTATION_MODES, Predictor, UNetGraph)

HERE = os.path.dirname(__file__)
PREDICTOR_HOME = os.path.join(HERE, "vendor", "sustain_ml_predictor")
UNET_MODELS_DIR = os.path.join(PREDICTOR_HOME, "unet_models")
DEFAULT_DEVICE = "xczu19eg-ffvb1517-2-i"
DEFAULT_DEVICE_DIR = os.path.join(PREDICTOR_HOME, DEFAULT_DEVICE)

_lock = threading.Lock()
_predictor = None   # DFKI's 9 surrogates, loaded once on the first FPGA request
_measured = {}      # (fps, implementation) -> U-Net files DFKI measured under that condition


def _get_predictor():
    global _predictor
    with _lock:
        if _predictor is None:
            _predictor = Predictor()
        return _predictor


def _measured_models(fps, implementation):
    """The U-Nets of unet_models/ that DFKI measured at that frame rate and implementation. A U-Net
    is missing from a condition when it did not fit on the FPGA under it (over-utilization)."""
    key = (int(fps), implementation)
    with _lock:
        if key not in _measured:
            path = os.path.join(DEFAULT_DEVICE_DIR, f"measurements_{key[0]}fps_{implementation}.jsonl")
            names = set()
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    names = {json.loads(line)["model_file"] for line in f if line.strip()}
            _measured[key] = names
        return _measured[key]


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def predict_latency_energy(onnx_model_path: str,
                           fps: int = DEFAULT_FPS,
                           implementation: str = DEFAULT_IMPLEMENTATION) -> dict:
    """
    Latency, power and energy of a U-Net on the FPGA board, predicted by the surrogates DFKI trained
    on their measurements of the U-Nets in unet_models/ (vendor/sustain_ml_predictor/predictors).

    The FPGA design of a U-Net depends on the frame rate it targets, fps (18 or 35), and on where it
    keeps the skip connections, implementation ("on_chip" or "off_chip" memory); by default 35 FPS
    and off-chip, the default of DFKI's predictor. The latency is the time per frame of that design
    (1 / achieved FPS).

    It returns the latency and the model's power and energy: the board's extra power while it runs
    the model (dynamic energy: board running minus board idle), as for every hardware in
    SustainML. The board's own power while running, idle included (total energy), is returned
    too, for reference, with the FPGA resources of the design.

    Raises ValueError for a U-Net outside the design space the surrogates were trained on (or not
    exported like the U-Nets of unet_models/), or for an unknown fps or implementation.
    """
    if int(fps) not in FPS_OPTIONS:
        raise ValueError(f"fps must be one of {FPS_OPTIONS}, got {fps!r}")
    if implementation not in IMPLEMENTATION_MODES:
        raise ValueError(f"implementation must be one of {IMPLEMENTATION_MODES}, got {implementation!r}")
    fps = int(fps)

    predictor = _get_predictor()
    unet = UNetGraph(onnx_model_path)
    pred = predictor.predict(unet, fps, implementation)

    # The surrogates return the units of the DFKI measurements they were trained on: latency in
    # seconds per frame and energy in joules per frame (total energy = runtime power [W] x latency
    # [s], dynamic energy = (runtime - idle power) [W] x latency [s])
    lat_s = pred["latency"]

    # Power of the model [W]: the board's extra power while it runs it, from its dynamic energy [J]
    # and latency [s]. The board's idle power (about 18-21 W in DFKI's measurements) is left out, as it
    # does not depend on the model. A surrogate can return a negative energy for a model unlike those
    # it learnt from: counted as 0
    energy_dynamic_J = max(pred["dynamic_energy"], 0.0)
    energy_board_J = max(pred["total_energy"], 0.0)
    power_w = energy_dynamic_J / lat_s if lat_s > 0.0 else 0.0
    power_board_runtime_w = energy_board_J / lat_s if lat_s > 0.0 else 0.0

    # Predictions outside the values a surrogate was trained on are less reliable
    outside_trained_range = []
    for target, value in pred.items():
        lo, hi = predictor.trained_range(target)
        if not lo <= value <= hi:
            outside_trained_range.append(target)

    in_catalogue = os.path.realpath(os.path.dirname(onnx_model_path)) == os.path.realpath(UNET_MODELS_DIR)
    measured = in_catalogue and os.path.basename(onnx_model_path) in _measured_models(fps, implementation)

    return {
        "device": DEFAULT_DEVICE,
        "fps": fps,
        "implementation": implementation,
        "latency_s": lat_s,
        "latency_h": lat_s / 3600.0,  # WP3 expects hours
        "power_w": power_w,
        "energy_J": energy_dynamic_J,
        "energy_Wh": energy_dynamic_J / 3600.0,
        "energy_dynamic_Wh": energy_dynamic_J / 3600.0,
        "power_board_runtime_w": power_board_runtime_w,
        "energy_board_runtime_Wh": energy_board_J / 3600.0,
        # FPGA resources of the design: BRAM36 tiles, LUTs, flip-flops and DSP blocks
        "fpga_resources": {t: pred[t] for t in ("brams", "luts", "ffs", "dsps")},
        # MFLOPs per joule of total and of dynamic energy
        "energy_efficiency_mflop_per_J": {"total": pred["total_energy_efficiency"],
                                          "dynamic": pred["dynamic_energy_efficiency"]},
        # Whether the U-Net is one of unet_models/ that DFKI measured under this fps and
        # implementation (it fitted on the FPGA); False for a U-Net outside the catalogue
        "measured_by_dfki": measured,
        "outside_trained_range": outside_trained_range,
        "provenance": {
            "source": "predicted by DFKI's predictors",
            "predictors_dir": os.path.relpath(str(predictor.dir), HERE),
            "predictors": sorted(pred),
            "condition": {"fps": fps, "implementation": implementation},
            "model_sha256": _hash_file(onnx_model_path),
        }
    }

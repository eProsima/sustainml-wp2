# Copyright 2025 SustainML Consortium
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

import os, hashlib
import numpy as np

from .vendor.sustain_ml_predictor.predictor import predict  # vendored DFKI code

HERE = os.path.dirname(__file__)
PREDICTOR_HOME = os.path.join(HERE, "vendor", "sustain_ml_predictor")
DEFAULT_DEVICE = "xczu19eg-ffvb1517-2-i"
DEFAULT_DEVICE_DIR = os.path.join(PREDICTOR_HOME, DEFAULT_DEVICE)

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def predict_latency_energy(onnx_model_path: str,
                          device: str = DEFAULT_DEVICE,
                          stats_file: str = None) -> dict:
    """
    A predictor function that uses:
      - predictor_model_latency.onnx
      - predictor_model_energy_dynamic.onnx
      - predictor_model_energy_board_runtime.onnx

    It returns latency, power_w (derived) and energy (board-runtime energy).
    """
    if stats_file is None:
        stats_file = os.path.join(PREDICTOR_HOME, device, "unet_models_stats.json")

    device_dir = DEFAULT_DEVICE_DIR if device == DEFAULT_DEVICE else os.path.join(PREDICTOR_HOME, device)

    lat_path = os.path.join(device_dir, "predictor_model_latency.onnx")
    edyn_path = os.path.join(device_dir, "predictor_model_energy_dynamic.onnx")
    eboard_path = os.path.join(device_dir, "predictor_model_energy_board_runtime.onnx")

    for p in (stats_file, lat_path, edyn_path, eboard_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing predictor asset: {p}")

    model_hash = _hash_file(onnx_model_path)

    # Predict latency
    lat_pred = predict(onnx_model_file=onnx_model_path,
                        models_stats_file=stats_file,
                        prediction_model_file=lat_path)

    # Predict dynamic energy
    edyn_pred = predict(onnx_model_file=onnx_model_path,
                        models_stats_file=stats_file,
                        prediction_model_file=edyn_path)

    # Predict board runtime energy
    eboard_pred = predict(onnx_model_file=onnx_model_path,
                        models_stats_file=stats_file,
                        prediction_model_file=eboard_path)

    lat_h = float(np.array(lat_pred).reshape(-1)[0])               # hours
    energy_dynamic_Wh = float(np.array(edyn_pred).reshape(-1)[0])  # Wh
    energy_board_Wh   = float(np.array(eboard_pred).reshape(-1)[0])# Wh

    # Derive average board power [W] from energy [Wh] and latency [h]
    if lat_h > 0.0:
        power_w = energy_board_Wh / lat_h
    else:
        power_w = 0.0

    return {
        "device": device,
        "latency_h": lat_h,
        "power_w": power_w,
        "energy_Wh": energy_board_Wh,
        "energy_dynamic_Wh": energy_dynamic_Wh,
        "energy_board_runtime_Wh": energy_board_Wh,
        "provenance": {
            "stats_file": os.path.relpath(stats_file, HERE),
            "predictors": ["latency", "energy_dynamic", "energy_board_runtime"],
            "model_sha256": model_hash,
            "assets_dir": os.path.relpath(device_dir, HERE),
        }
    }

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
"""Write the architecture of every vendored U-Net to unet_models_info.jsonl.

The UI lists the U-Nets of the FPGA and draws their graph from this file, one line per model:
model_file, input_channels, input_size, initial_channels, depth, kernel_sizes, Mflops and Mparams.
DFKI's predictor package does not ship it, so it is generated from the ONNX files with the
predictor's own parser (UNetGraph), and the FLOPs and parameters are checked against DFKI's
measurements. Run it again whenever the vendored predictor is updated:

    python3 hw_provider_fpga/generate_unet_models_info.py
"""

import json
import sys
from pathlib import Path

PREDICTOR_HOME = Path(__file__).resolve().parent / "vendor" / "sustain_ml_predictor"
DEVICE_DIR = PREDICTOR_HOME / "xczu19eg-ffvb1517-2-i"
OUTPUT = DEVICE_DIR / "unet_models_info.jsonl"

sys.path.insert(0, str(PREDICTOR_HOME))
from predictor import UNetGraph  # noqa: E402


def measured_counts():
    """model_file -> (MFLOPs, parameters) from DFKI's measurement files."""
    counts = {}
    for path in sorted(DEVICE_DIR.glob("measurements_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                counts.setdefault(m["model_file"], (float(m["mflops"]), int(m["params"])))
    return counts


def main():
    measured = measured_counts()
    lines = []
    for path in sorted((PREDICTOR_HOME / "unet_models").glob("*.onnx")):
        unet = UNetGraph(path)
        spec = unet.spec
        mflops = round(unet.total_flops / 1e6, 6)
        params = int(round(unet.total_params))
        if path.name in measured:
            m_mflops, m_params = measured[path.name]
            if abs(mflops - m_mflops) > 1e-6 or params != m_params:
                raise SystemExit(f"{path.name}: {mflops} MFLOPs and {params} parameters in the ONNX "
                                 f"graph, {m_mflops} and {m_params} in the measurements")
        lines.append(json.dumps({
            "model_file": path.name,
            "input_channels": spec["input_channels"],
            "input_size": spec["input_size"],
            "initial_channels": spec["initial_channels"],
            "depth": spec["depth"],
            "kernel_sizes": spec["kernel_sizes"],
            "Mflops": mflops,
            "Mparams": round(params / 1e6, 6),
        }))
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUTPUT}: {len(lines)} U-Nets ({sum(1 for l in lines if json.loads(l)['model_file'] in measured)} "
          f"checked against the measurements)")


if __name__ == "__main__":
    main()

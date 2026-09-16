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
"""Puts the values measured on the Axelera board, and the memory of the models, in yolo_models.json.

  --latency CSV    one input alone (measure_single_yolo.sh): measured.latency_s = 1 / mean system FPS
  --full-load CSV  all cores (measure_all_yolo.sh): measured.full_load_fps = mean system FPS
  --memory         measured weights_mb, working_mb, memory_mb and parameters, computed from each
                   model's ONNX file as for any ONNX model (model_memory.onnx_memory_mb). The file is
                   taken from --onnx-dir/<model>/*.onnx (the SDK's ~/.cache/axelera/weights), else
                   downloaded from onnx_url
  --models M ...   only these models (default: all)

The CSVs have the columns model,task,input,run,dev_fps,host_fps,system_fps,... (one row per run).
Usage, from sustainml-wp2: python3 hw_provider_axelera/import_measurements.py --latency single.csv
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import model_memory  # noqa: E402

CATALOG_FILE = os.path.join(HERE, "yolo_models.json")


def mean_system_fps(csv_path):
    runs = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("system_fps"):
                runs.setdefault(row["model"], []).append(float(row["system_fps"]))
    return {model: sum(fps) / len(fps) for model, fps in runs.items()}


def onnx_file(model, onnx_dir, tmp):
    if onnx_dir:
        found = sorted(glob.glob(os.path.join(onnx_dir, model["name"], "*.onnx")))
        if found:
            return found[0]
    path = os.path.join(tmp, model["name"] + ".onnx")
    urllib.request.urlretrieve(model["onnx_url"], path)
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    if model.get("onnx_md5") and md5 != model["onnx_md5"]:
        raise ValueError(f"{model['onnx_url']}: MD5 {md5}, expected {model['onnx_md5']}")
    return path


def parameters(path):
    import onnx
    total = 0
    for tensor in onnx.load(path, load_external_data=False).graph.initializer:
        count = 1
        for dim in tensor.dims:
            count *= dim
        total += count
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latency")
    parser.add_argument("--full-load")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--onnx-dir")
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()

    with open(CATALOG_FILE) as f:
        catalog = json.load(f)
    models = [m for m in catalog["models"] if not args.models or m["name"] in args.models]

    for key, path in (("latency_s", args.latency), ("full_load_fps", args.full_load)):
        if not path:
            continue
        for name, fps in mean_system_fps(path).items():
            model = next((m for m in models if m["name"] == name), None)
            if model is None:
                print(f"{name}: not in the catalog, skipped")
                continue
            model["measured"][key] = round(1.0 / fps, 6) if key == "latency_s" else round(fps, 1)
            print(f"{name}: {key} = {model['measured'][key]}")

    if args.memory:
        with tempfile.TemporaryDirectory() as tmp:
            for model in models:
                path = onnx_file(model, args.onnx_dir, tmp)
                hw = [int(v) for v in model["input_resolution"].split("x")]
                weights, working = model_memory.onnx_memory_mb(path, hw)
                model["measured"].update({"weights_mb": weights, "working_mb": working,
                                          "memory_mb": round(weights + working, 2),
                                          "parameters": parameters(path)})
                print(f"{model['name']}: {weights} MB weights + {working} MB working memory")
                if path.startswith(tmp):
                    os.remove(path)

    with open(CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()

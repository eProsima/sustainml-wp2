# U-Net FPGA surrogate predictor

Predicts the FPGA resources, latency and energy of a U-Net from its ONNX file, for the
`xczu19eg-ffvb1517-2-i` part at 18 or 35 FPS with on-chip or off-chip memory
(default: **35 FPS, off-chip**).

```text
├── predictor.py                    # the predictor (standalone module and command line)
├── demo.py                         # compare predictions with the measurements
├── predictors/                     # one surrogate per target (ONNX, output in the target's units)
│   ├── manifest.json               # model, normalization, inputs, units, CV scores
│   └── feature_stats.json          # encoding statistics used by predictor.py
├── unet_models/                    # the 394 measured U-Nets, unet_model_<Model_id>.onnx
├── xczu19eg-ffvb1517-2-i/          # measurements (ground truth), one file per condition
│   └── measurements_<18|35>fps_<on|off>_chip.jsonl
├── demo_results/                   # output of demo.py
└── unet_model.png                  # operation graph of unet_model_760.onnx (depth 4)
```

Needs numpy, onnx and onnxruntime. `demo.py` also needs scipy, matplotlib and tqdm.

## Targets

| Target | Units |
|---|---|
| `brams` | BRAM36 tiles (RAMB36 + 0.5 x RAMB18) |
| `luts`, `ffs`, `dsps` | LUTs, flip-flops, DSP blocks |
| `latency` | s per frame (1 / achieved FPS) |
| `total_energy` | J per frame (runtime power x latency) |
| `dynamic_energy` | J per frame ((runtime - idle power) x latency) |
| `total_energy_efficiency` | MFLOP/J (MFLOPs / latency / runtime power) |
| `dynamic_energy_efficiency` | MFLOP/J (MFLOPs / latency / (runtime - idle power)) |

Each target has its own surrogate, a graph neural network chosen by cross-validation SMAPE.

## Predict from Python

`predictor.py` is a single module. Put this folder on `sys.path` (or copy the module along
with `predictors/`). It finds `predictors/` next to itself from any working directory.
Model paths are ordinary paths, so the examples below assume you run from this folder.

```python
from predictor import Predictor, UNetGraph

predictor = Predictor()                     # loads all 9 surrogates; do this once
predictor.predict("unet_models/unet_model_270.onnx")                # 35 FPS, off-chip
# -> {"brams": 197.46, "luts": 51984.0, ..., "latency": 0.0234, ...}  (floats, in the units above)
predictor.predict("unet_models/unet_model_270.onnx", fps=18, implementation="on_chip")

unet = UNetGraph("unet_models/unet_model_270.onnx")   # read a file once, predict several conditions
for fps in (18, 35):
    for implementation in ("on_chip", "off_chip"):
        print(fps, implementation, predictor.predict(unet, fps, implementation))
```

- `Predictor(targets=["latency", "dynamic_energy"])` loads only those surrogates.
- `predictor.units(target)` returns the units.
- `predictor.trained_range(target)` returns the smallest and largest value the surrogate
  was trained on; treat predictions outside that range with suspicion.
- `fps` must be 18 or 35 and `implementation` must be `"on_chip"` or `"off_chip"`. A file
  outside the design space raises `ValueError`.

## Command line

```bash
python predictor.py --model_file unet_models/unet_model_270.onnx
python predictor.py --model_file unet_models/unet_model_270.onnx --metric latency dynamic_energy --fps 18 --implementation on_chip
```

`demo.py` compares the predictions with the measurements of one condition:

```bash
python demo.py                                  # every measured U-Net, 35 FPS, off-chip
python demo.py --fps 18 --implementation on_chip
python demo.py --model_file 270                 # one U-Net (Model_id, file name or path)
```

For all U-Nets, the demo prints SMAPE, R², Spearman and Kendall per target and writes the
predictions and a predicted-vs-measured plot to `demo_results/`. With `--model_file`, it
prints that U-Net's measured value, prediction and relative error per target. The U-Net
must be measured under the chosen condition.

## Measurement files

One line per U-Net that was measured under that condition:

- `model_file`
- `uid`: `<Model_id>_<fps>_<0 on-chip | 1 off-chip>`
- `status`
- `mflops`: the FLOP count, in millions, that the efficiency targets use
- `params`: weights and biases
- the 9 targets
- `power_runtime` and `power_idle` in W

A U-Net appears only under the conditions where it fit on the FPGA. Measurements that
failed in Vivado (over-utilization) are left out of their condition. 16 architectures whose
runtime power was at or below idle power were left out entirely. Every line has all 9
targets.

## Limits

- **Design space.** Depth 1-4, initial channels 8/16/32/64, input 3 x 64..512 x 64..512,
  kernel 3 or 5 per level. `predictor.py` rejects other models.
- **The ONNX export must match.** The surrogates encode the graph operation by operation.
  A U-Net that is not one of the files in `unet_models/` must be exported the same way:
  `proxy_training/model.py`'s UNet, the TorchScript exporter, opset 13 and shapes
  inferred. Another exporter, opset or a graph simplifier changes the graph and the
  predictions.

  ```python
  torch.onnx.export(model.eval(), torch.randn(1, 3, size, size), path, dynamo=False,
                    training=torch.onnx.TrainingMode.EVAL, input_names=["input_data"],
                    output_names=["output_prediction"], do_constant_folding=True,
                    opset_version=13)
  onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
  ```

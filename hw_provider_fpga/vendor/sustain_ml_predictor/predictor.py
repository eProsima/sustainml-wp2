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

"""Predict the FPGA resources, latency and energy of a U-Net ONNX model.

    python predictor.py --model_file unet_models/unet_model_270.onnx
                        [--metric latency [dynamic_energy ...]]
                        [--fps {18,35}] [--implementation {on_chip,off_chip}]

By default every target is predicted for 35 FPS with off-chip memory.

The surrogates in predictors/ (one ONNX file per target, described in
predictors/manifest.json) were trained on xczu19eg-ffvb1517-2-i measurements of the U-Net
design space of proxy_training/model_specs.py: depth 1-4, initial channels 8/16/32/64,
input 3 x 64..512 x 64..512 and a kernel size of 3 or 5 per level. Models outside that
space are rejected.

A U-Net file is encoded exactly as the training encodings were
(proxy_training/generate_unet_predictor_inputs_updated.py): per-operation features and
connectivity from the ONNX graph, the architecture vector and total FLOPs and parameters
as global features, and the condition flags [fps_18, fps_35, on_chip, off_chip], normalized
with predictors/feature_stats.json. The graph must therefore come from the same export as
in training: proxy_training/model.py's UNet exported with the TorchScript exporter at
opset 13 and shapes inferred, like the files in unet_models/. Another exporter or a graph
simplifier changes the operations and with them the predictions.

Needs numpy, onnx and onnxruntime.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

HERE = Path(__file__).resolve().parent
DEFAULT_PREDICTORS = HERE / "predictors"

FPS_OPTIONS = (18, 35)
IMPLEMENTATION_MODES = ("on_chip", "off_chip")
DEFAULT_FPS = 35
DEFAULT_IMPLEMENTATION = "off_chip"

# Encoding layout of generate_unet_predictor_inputs_updated.py.
REAL_OP_TYPES = ("Concat", "Conv", "ConvTranspose", "MaxPool", "Relu")
GNN_CATEGORIES = ("None", "Global") + REAL_OP_TYPES
GNN_MAX_NODES = 50  # one global node + at most 49 operations
DEPTH_OPTIONS = (1, 2, 3, 4)
INITIAL_CHANNEL_OPTIONS = (8, 16, 32, 64)
INPUT_SIZE_OPTIONS = (64, 128, 256, 512)
INPUT_CHANNEL_OPTIONS = (3,)
KERNEL_OPTIONS = (3, 5)
MAX_KERNEL_POSITIONS = 5  # depth + 1 <= 5


def one_hot(value, options):
    return [1.0 if value == option else 0.0 for option in options]


def condition_vector(fps, implementation):
    """[fps_18, fps_35, on_chip, off_chip]."""
    if int(fps) not in FPS_OPTIONS:
        raise ValueError(f"fps must be one of {FPS_OPTIONS}, got {fps!r}")
    if implementation not in IMPLEMENTATION_MODES:
        raise ValueError(f"implementation must be one of {IMPLEMENTATION_MODES}, "
                         f"got {implementation!r}")
    return np.asarray(one_hot(int(fps), FPS_OPTIONS)
                      + one_hot(implementation, IMPLEMENTATION_MODES), dtype=np.float32)


# ----------------------------------------------------------------------------- ONNX graph

def _attribute(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return None


def _value_shapes(graph):
    """Tensor name -> shape, for tensors whose dimensions are all known."""
    shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = vi.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
            shapes[vi.name] = [int(d.dim_value) for d in dims]
    return shapes


def _shapes_4d(names, shapes):
    return [shapes[n] for n in names if n in shapes and len(shapes[n]) == 4]


def _square(shape, what):
    if shape[2] != shape[3]:
        raise ValueError(f"{what} is not square: {shape}")
    return float(shape[2])


def _kernel_size(node, initializers):
    kernel = _attribute(node, "kernel_shape")
    if kernel is None and node.op_type in ("Conv", "ConvTranspose"):
        weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
        if weight is not None and len(weight.dims) >= 4:
            kernel = list(weight.dims[-2:])
    if kernel is None:
        return 0.0
    kernel = [int(k) for k in kernel]
    if len(kernel) == 2 and kernel[0] != kernel[1]:
        raise ValueError(f"{node.op_type} has a non-square kernel {kernel}")
    return float(kernel[0])


def _flops(node, inputs, output, initializers):
    """Per-operation workload, with the convention of the training encodings:
    multiply-accumulates for Conv/ConvTranspose, output elements for Relu, comparisons
    for MaxPool, moved input elements for Concat."""
    if node.op_type in ("Conv", "ConvTranspose"):
        k = int(_kernel_size(node, initializers))
        groups = int(_attribute(node, "group") or 1)
        n, cin = inputs[0][:2]
        _, cout, hout, wout = output
        return float(n * cout * hout * wout * (cin // groups) * k * k)
    if node.op_type == "Relu":
        return float(np.prod(output))
    if node.op_type == "MaxPool":
        k = int(_kernel_size(node, initializers))
        return float(np.prod(output) * max(k * k - 1, 0))
    return float(sum(np.prod(s) for s in inputs))  # Concat


class UNetGraph:
    """What the surrogates read from a U-Net ONNX file: one row of nine numerical
    features per operation [FLOPs, parameters, C_in, C_out, S_in, S_out, kernel,
    in-degree, out-degree], the producer -> consumer edges with the size of the tensor
    they carry and whether they are a skip connection, and the architecture."""

    def __init__(self, path):
        self.path = Path(path)
        model = onnx.load(str(path))
        if not model.graph.value_info:
            model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph
        initializers = {t.name: t for t in graph.initializer}
        shapes = _value_shapes(graph)

        unexpected = sorted({n.op_type for n in graph.node} - set(REAL_OP_TYPES))
        if unexpected:
            raise ValueError(f"{self.path.name}: operations {unexpected} are not part of the U-Net design space (only {', '.join(REAL_OP_TYPES)})")

        self.op_types = []
        rows = []
        producer = {}
        for i, node in enumerate(graph.node):
            inputs = _shapes_4d(node.input, shapes)
            outputs = _shapes_4d(node.output, shapes)
            if not inputs or not outputs:
                raise ValueError(f"{self.path.name}: no 4-D tensor shapes for {node.op_type} "
                                 f"node {node.name!r}")
            output = outputs[0]
            if node.op_type == "Concat":
                cin = float(sum(s[1] for s in inputs))
                sizes = {_square(s, "Concat input") for s in inputs}
                if len(sizes) != 1:
                    raise ValueError(f"Concat inputs have different sizes: {inputs}")
                sin = sizes.pop()
            else:
                cin, sin = float(inputs[0][1]), _square(inputs[0], f"{node.op_type} input")
            params = float(sum(int(np.prod(list(initializers[n].dims)))
                               for n in node.input if n in initializers))
            rows.append([_flops(node, inputs, output, initializers), params, cin,
                         float(output[1]), sin, _square(output, f"{node.op_type} output"),
                         _kernel_size(node, initializers), 0.0, 0.0])
            self.op_types.append(node.op_type)
            for name in node.output:
                producer[name] = i

        src, dst, volume, skip = [], [], [], []
        for j, node in enumerate(graph.node):
            for name in node.input:
                i = producer.get(name)
                if i is None:
                    continue
                shape = shapes.get(name)
                src.append(i)
                dst.append(j)
                volume.append(float(np.prod(shape[1:])) if shape and len(shape) == 4 else 0.0)
                # A decoder Concat gets the ConvTranspose output and an encoder skip.
                skip.append(float(node.op_type == "Concat"
                                  and graph.node[i].op_type != "ConvTranspose"))

        self.numeric = np.asarray(rows, dtype=np.float64).reshape(-1, 9)
        for i, j in zip(src, dst):
            self.numeric[i, 8] += 1.0  # out-degree
            self.numeric[j, 7] += 1.0  # in-degree
        self.edge_index = np.asarray([src, dst], dtype=np.int64).reshape(2, -1)
        self.edge_volume = np.asarray(volume, dtype=np.float64)
        self.edge_skip = np.asarray(skip, dtype=np.float32)
        self.total_flops = float(self.numeric[:, 0].sum())
        self.total_params = float(self.numeric[:, 1].sum())
        self.spec = self._architecture(graph, initializers, shapes)

    def _architecture(self, graph, initializers, shapes):
        """depth, kernel sizes, initial channels and input shape of the U-Net."""
        data_inputs = [vi.name for vi in graph.input if vi.name not in initializers]
        if len(data_inputs) != 1 or data_inputs[0] not in shapes:
            raise ValueError(f"{self.path.name}: expected one input of known shape")
        _, input_channels, input_size, _ = shapes[data_inputs[0]]

        # In execution order: two Convs per encoder level, two in the middle, two per
        # decoder level and the final 1x1 Conv; one ConvTranspose per decoder level.
        depth = self.op_types.count("ConvTranspose")
        convs = [n for n in graph.node if n.op_type == "Conv"]
        if len(convs) != 4 * depth + 3:
            raise ValueError(f"{self.path.name}: {len(convs)} Conv nodes for depth {depth}, "
                             "not a U-Net of proxy_training/model.py")
        kernels = [int(_kernel_size(c, initializers)) for c in convs]
        kernel_sizes = kernels[0:2 * depth + 2:2]
        if kernels[1:2 * depth + 2:2] != kernel_sizes:
            raise ValueError(f"{self.path.name}: the two Convs of a level differ in kernel size")
        spec = {
            "depth": depth,
            "kernel_sizes": kernel_sizes,
            "initial_channels": int(initializers[convs[0].input[1]].dims[0]),
            "input_size": int(input_size),
            "input_channels": int(input_channels),
        }
        allowed = {"depth": DEPTH_OPTIONS, "initial_channels": INITIAL_CHANNEL_OPTIONS,
                   "input_size": INPUT_SIZE_OPTIONS, "input_channels": INPUT_CHANNEL_OPTIONS}
        outside = [f"{k}={spec[k]}" for k, options in allowed.items() if spec[k] not in options]
        outside += [f"kernel size {k}" for k in kernel_sizes if k not in KERNEL_OPTIONS]
        if outside:
            raise ValueError(f"{self.path.name}: outside the design space the surrogates were "
                             f"trained on ({', '.join(outside)})")
        if len(self.op_types) > GNN_MAX_NODES - 1:
            raise ValueError(f"{self.path.name}: {len(self.op_types)} operations, the "
                             f"surrogates take at most {GNN_MAX_NODES - 1}")
        return spec

    def architecture_vector(self):
        """22-D one-hot architecture encoding: depth, initial channels, input size and
        two entries per kernel position (zero-padded to 5 positions)."""
        s = self.spec
        kernels = []
        for position in range(MAX_KERNEL_POSITIONS):
            kernels += (one_hot(s["kernel_sizes"][position], KERNEL_OPTIONS)
                        if position < len(s["kernel_sizes"]) else [0.0, 0.0])
        return np.asarray(one_hot(s["depth"], DEPTH_OPTIONS)
                          + one_hot(s["initial_channels"], INITIAL_CHANNEL_OPTIONS)
                          + one_hot(s["input_size"], INPUT_SIZE_OPTIONS) + kernels,
                          dtype=np.float32)

    def describe(self):
        s = self.spec
        return (f"{self.path.name}: depth {s['depth']}, initial channels "
                f"{s['initial_channels']}, input {s['input_channels']}x{s['input_size']}x"
                f"{s['input_size']}, kernels {s['kernel_sizes']}, {len(self.op_types)} "
                f"operations, {self.total_params / 1e6:.3g} M parameters")


# ----------------------------------------------------------------------------- encoding

class FeatureStats:
    """z-score statistics of the training encodings (predictors/feature_stats.json)."""

    def __init__(self, path):
        stats = json.loads(Path(path).read_text(encoding="utf-8"))
        get = lambda key: (np.asarray(stats[key]["mean"], dtype=np.float64),  # noqa: E731
                           np.asarray(stats[key]["std"], dtype=np.float64))
        self.node = get("node_stats")
        self.mlp = get("mlp_stats")
        self.edge_volume = get("edge_volume_stats")

    @staticmethod
    def _z(values, mean_std):
        mean, std = mean_std
        return (values - mean) / std

    def node_features(self, numeric):
        """FLOPs and parameters log1p, channels and sizes log2, the rest raw; z-scored."""
        out = numeric.copy()
        out[:, 0:2] = np.log1p(np.maximum(out[:, 0:2], 0.0))
        if np.any(out[:, 2:6] <= 0):
            raise ValueError("channel counts and sizes must be positive")
        out[:, 2:6] = np.log2(out[:, 2:6])
        return self._z(out, self.node).astype(np.float32)

    def global_features(self, total_flops, total_params):
        values = np.log1p(np.maximum(np.asarray([total_flops, total_params]), 0.0))
        return self._z(values, self.mlp).astype(np.float32)

    def edge_volumes(self, volumes):
        return self._z(np.log1p(np.maximum(volumes, 0.0)), self.edge_volume).astype(np.float32)


def encode(unet, stats, fps=DEFAULT_FPS, implementation=DEFAULT_IMPLEMENTATION):
    """The surrogate inputs of one U-Net under one condition, keyed like the training
    encodings file (manifest "encoding_key")."""
    condition = condition_vector(fps, implementation)
    numeric = stats.node_features(unet.numeric)
    n = len(unet.op_types)

    gnn_x = np.zeros((GNN_MAX_NODES, len(GNN_CATEGORIES) + 9 + 4), dtype=np.float32)
    gnn_x[0, GNN_CATEGORIES.index("Global")] = 1.0  # global node: no numeric features
    gnn_x[0, -4:] = condition
    for i, op in enumerate(unet.op_types, start=1):
        gnn_x[i, GNN_CATEGORIES.index(op)] = 1.0
    gnn_x[1:n + 1, len(GNN_CATEGORIES):-4] = numeric
    gnn_x[1:n + 1, -4:] = condition
    gnn_x[n + 1:, GNN_CATEGORIES.index("None")] = 1.0  # padding rows stay zero otherwise

    adjacency = np.zeros((GNN_MAX_NODES, GNN_MAX_NODES), dtype=np.float32)
    adjacency[unet.edge_index[0] + 1, unet.edge_index[1] + 1] = 1.0
    adjacency[0, :] = 1.0  # the global node reaches every position
    np.fill_diagonal(adjacency, 1.0)

    gat_x = np.zeros((n, len(REAL_OP_TYPES) + 9 + 4), dtype=np.float32)
    for i, op in enumerate(unet.op_types):
        gat_x[i, REAL_OP_TYPES.index(op)] = 1.0
    gat_x[:, len(REAL_OP_TYPES):-4] = numeric
    gat_x[:, -4:] = condition

    return {
        "mlp": np.concatenate([unet.architecture_vector(),
                               stats.global_features(unet.total_flops, unet.total_params),
                               condition]).astype(np.float32),
        "gnn.x": gnn_x,
        "gnn.adj_matrix": adjacency,
        "gatv2.x": gat_x,
        "gatv2.edge_index": unet.edge_index,
        "gatv2.edge_attr": np.stack([stats.edge_volumes(unet.edge_volume), unet.edge_skip],
                                    axis=1).reshape(-1, 2).astype(np.float32),
    }


# ----------------------------------------------------------------------------- predictor

class Predictor:
    """The surrogates of predictors/, one ONNX Runtime session per target."""

    def __init__(self, predictors_dir=DEFAULT_PREDICTORS, targets=None):
        self.dir = Path(predictors_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        available = self.manifest["targets"]
        targets = list(targets or available)
        unknown = [t for t in targets if t not in available]
        if unknown:
            raise ValueError(f"no predictor for {unknown}; available: {', '.join(available)}")
        self.targets = {t: available[t] for t in targets}
        self.stats = FeatureStats(self.dir / self.manifest["feature_stats"])
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self.sessions = {
            t: onnxruntime.InferenceSession(str(self.dir / e["file"]), options,
                                            providers=["CPUExecutionProvider"])
            for t, e in self.targets.items()
        }

    @property
    def fpga_part(self):
        return ", ".join(self.manifest.get("fpga_part") or ["unknown FPGA"])

    def units(self, target):
        return self.targets[target]["units"]

    def trained_range(self, target):
        """Smallest and largest target value the predictor was trained on."""
        norm = self.targets[target]["normalizer"]
        lo, hi = norm["minimum"], norm["maximum"]
        if norm["method"].startswith("log1p"):
            lo, hi = float(np.expm1(lo)), float(np.expm1(hi))
        return lo, hi

    def predict(self, model, fps=DEFAULT_FPS, implementation=DEFAULT_IMPLEMENTATION):
        """{target: prediction in the target's units} for a U-Net ONNX path or UNetGraph."""
        unet = model if isinstance(model, UNetGraph) else UNetGraph(model)
        encoded = encode(unet, self.stats, fps, implementation)
        predictions = {}
        for target, entry in self.targets.items():
            feeds = {}
            for spec in entry["inputs"]:
                value = encoded[spec["encoding_key"]].astype(spec["dtype"])
                feeds[spec["name"]] = value if entry["one_graph_per_call"] else value[None]
            output = self.sessions[target].run([entry["output"]], feeds)[0]
            predictions[target] = float(np.asarray(output).reshape(-1)[0])
        return predictions


def predict(model_file, targets=None, fps=DEFAULT_FPS, implementation=DEFAULT_IMPLEMENTATION,
            predictors_dir=DEFAULT_PREDICTORS):
    """One-off prediction; build a Predictor once to predict many models."""
    return Predictor(predictors_dir, targets).predict(model_file, fps, implementation)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_file", required=True, help="U-Net ONNX file")
    parser.add_argument("--metric", nargs="+", default=None, help="targets to predict (default: all in the predictors manifest)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, choices=FPS_OPTIONS, help="target frame rate (default: %(default)s)")
    parser.add_argument("--implementation", default=DEFAULT_IMPLEMENTATION, choices=IMPLEMENTATION_MODES, help="memory implementation (default: %(default)s)")
    parser.add_argument("--predictors_dir", default=DEFAULT_PREDICTORS, help="directory with manifest.json and the predictors (default: %(default)s)")
    args = parser.parse_args()

    try:
        predictor = Predictor(args.predictors_dir, args.metric)
        unet = UNetGraph(args.model_file)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    predictions = predictor.predict(unet, args.fps, args.implementation)

    print(unet.describe())
    print(f"Predictions for {predictor.fpga_part} at {args.fps} FPS, "
          f"{args.implementation.replace('_', '-')} memory:")
    width = max(len(t) for t in predictions)
    for target, value in predictions.items():
        lo, hi = predictor.trained_range(target)
        note = "" if lo <= value <= hi else f"  (outside the trained range {lo:.4g} to {hi:.4g})"
        print(f"  {target:<{width}}  {value:12.5g}  {predictor.units(target)}{note}")


if __name__ == "__main__":
    main()

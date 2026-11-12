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
"""Memory footprint of a model, the same way for every hardware: weights plus working memory.

  weights:        the model's parameters and buffers
  working memory: the tensors processing one input creates that are alive at the same time, at
                  their peak, for the largest input the model accepts (a model of fixed input size,
                  as a CNN, has only that one)

Every floating-point value is counted at 32 bits (FLOAT_BYTES), whatever precision the model is
stored in, loaded in or runs at on the hardware, so that models compare; other values (integer
indices...) at their own size. In MB of 1024 x 1024 bytes.

A Hugging Face model's is computed from its config without its weights (config_memory_mb), in a
fresh process: the UPMEM simulator used by the HW node replaces PyTorch's layers for the whole
process, and the memory must be that of PyTorch's own layers. An ONNX model's from its file
(onnx_memory_mb).
"""

import multiprocessing
import queue as queue_module
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

MB = 1024 * 1024
FLOAT_BYTES = 4


# Bytes of the tensors a forward pass creates that are alive at the same time: their peak
class _PeakTensorMemory(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.current = 0
        self.peak = 0
        self.storages = {}

    def _release(self, key):
        entry = self.storages.get(key)
        if entry is not None:
            entry[1] -= 1
            if entry[1] == 0:
                self.current -= entry[0]
                del self.storages[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Meta tensors have no values: data-dependent checks (e.g. "any inf in float16?") get 0/False
        if func is torch.ops.aten.is_nonzero.default:
            return False
        if func is torch.ops.aten._local_scalar_dense.default:
            return False if args[0].dtype == torch.bool else 0
        out = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(out)[0]:
            if isinstance(tensor, torch.Tensor):
                key = tensor.untyped_storage()._cdata
                if key not in self.storages:
                    self.storages[key] = [tensor.untyped_storage().nbytes(), 0]
                    self.current += self.storages[key][0]
                    self.peak = max(self.peak, self.current)
                self.storages[key][1] += 1
                weakref.finalize(tensor, self._release, key)
        return out


class _MetaAutocast(torch.autocast):
    # Some models open torch.autocast for their tensors' device: "meta" is not an autocast device
    def __init__(self, device_type, *args, **kwargs):
        super().__init__("cpu" if device_type == "meta" else device_type, *args, **kwargs)


# The model built from its config without weights (on the meta device), in float32 so that its
# working memory counts floating-point values at FLOAT_BYTES
def _model_without_weights(model_name, hf_token):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    with init_empty_weights(include_buffers=True):
        if getattr(config, "is_encoder_decoder", False):
            model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)
        else:
            try:
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)
            except Exception:
                model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)
    model = model.to(torch.float32).eval()
    model.tie_weights()
    return model, config


def _tensor_bytes(tensor):
    return tensor.numel() * (FLOAT_BYTES if tensor.is_floating_point() else tensor.element_size())


# Weights (MB) of a PyTorch model, loaded or built without weights: parameters (tied weights
# counted once) plus buffers
def weights_mb(model):
    params = sum(_tensor_bytes(p) for p in model.parameters())
    buffers = sum(_tensor_bytes(b) for b in model.buffers())
    return round((params + buffers) / MB, 2)


# Memory (MB) the model needs besides its weights to process the longest input it accepts:
# activations, attention cache and outputs, at their peak. Computed by running its forward pass on
# meta tensors (no data, no computation), which allocates exactly what the real pass does
def _working_memory_mb(model, config):
    length = None
    for key in ("max_position_embeddings", "n_positions"):
        value = getattr(config, key, None)
        if isinstance(value, int) and 0 < value <= 1_000_000:
            length = value
            break
    if length is None:
        print("[WARN] Unknown maximum input length: working memory not computed")
        return 0.0
    ids = torch.ones((1, length), dtype=torch.long, device="meta")
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if getattr(config, "is_encoder_decoder", False):
        inputs["decoder_input_ids"] = ids
    tracker = _PeakTensorMemory()
    real_autocast = torch.autocast
    torch.autocast = _MetaAutocast
    try:
        with torch.inference_mode(), tracker:
            outputs = model(**inputs, use_cache=True)
            del outputs
    finally:
        torch.autocast = real_autocast
    return round(tracker.peak / MB, 2)


def _compute(model_name, hf_token, results):
    try:
        model, config = _model_without_weights(model_name, hf_token)
        results.put((weights_mb(model), _working_memory_mb(model, config)))
    except Exception as e:
        results.put(RuntimeError(f"{type(e).__name__}: {e}"))


# (weights, working memory) in MB of a Hugging Face model, computed in a fresh process
def config_memory_mb(model_name, hf_token=None, timeout=300):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=_compute, args=(model_name, hf_token, results))
    process.start()
    try:
        result = results.get(timeout=timeout)
    except queue_module.Empty:
        result = TimeoutError(f"no result within {timeout} s")
    finally:
        process.join(5)
        if process.is_alive():
            process.terminate()
    if isinstance(result, Exception):
        raise result
    return result


def _onnx_value_bytes(data_type, dims):
    from onnx import TensorProto, helper
    count = 1
    for dim in dims:
        count *= dim
    if data_type in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE):
        return count * FLOAT_BYTES
    return count * helper.tensor_dtype_to_np_dtype(data_type).itemsize


# (weights, working memory) in MB of an ONNX model. The working memory is that of one input: a
# model accepting any image size (dynamic height and width) is given the one it runs at in
# input_hw, (height, width). The size of every intermediate result comes from ONNX shape
# inference or, for those it cannot tell, from running the model once on that input; each one is
# alive from the node that produces it to the last node that uses it
def onnx_memory_mb(model_path, input_hw=None):
    import numpy as np
    import onnx
    from onnx import helper, shape_inference

    model = onnx.load(model_path, load_external_data=False)
    initializers = {t.name for t in model.graph.initializer}
    weights = sum(_onnx_value_bytes(t.data_type, t.dims) for t in model.graph.initializer)

    # One input, at the given size where the model accepts any
    inputs = [v for v in model.graph.input if v.name not in initializers]
    for value in inputs:
        dims = value.type.tensor_type.shape.dim
        for i, dim in enumerate(dims):
            if dim.dim_value > 0:
                continue
            if i == 0:
                dim.dim_value = 1
            elif input_hw and len(dims) == 4 and i >= 2:
                dim.dim_value = int(input_hw[i - 2])
            else:
                print(f"[WARN] {model_path}: unknown size of input {value.name}: working memory not computed")
                return round(weights / MB, 2), 0.0

    model = shape_inference.infer_shapes(model, data_prop=True)
    sizes = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor = value.type.tensor_type
        if tensor.HasField("shape") and all(d.dim_value > 0 for d in tensor.shape.dim):
            sizes[value.name] = _onnx_value_bytes(tensor.elem_type, [d.dim_value for d in tensor.shape.dim])

    unknown = [n for node in model.graph.node for n in node.output if n and n not in sizes]
    if unknown:
        import onnxruntime
        probe = onnx.ModelProto()
        probe.CopyFrom(model)
        known_outputs = {o.name for o in probe.graph.output}
        probe.graph.output.extend(helper.make_empty_tensor_value_info(n) for n in unknown if n not in known_outputs)
        session = onnxruntime.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
        feed = {}
        for value in inputs:
            tensor = value.type.tensor_type
            feed[value.name] = np.zeros([d.dim_value for d in tensor.shape.dim],
                                        dtype=helper.tensor_dtype_to_np_dtype(tensor.elem_type))
        names = [o.name for o in session.get_outputs()]
        for name, array in zip(names, session.run(names, feed)):
            if name not in sizes:
                sizes[name] = _onnx_value_bytes(helper.np_dtype_to_tensor_dtype(array.dtype), array.shape)

    nodes = model.graph.node
    first = {v.name: 0 for v in inputs}
    last = {}
    for i, node in enumerate(nodes):
        for name in node.input:
            if name and name not in initializers:
                last[name] = i
        for name in node.output:
            if name:
                first.setdefault(name, i)
    for value in model.graph.output:
        last[value.name] = len(nodes)
    alive = [0] * (len(nodes) + 2)
    for name, start in first.items():
        alive[start] += sizes.get(name, 0)
        alive[max(last.get(name, start), start) + 1] -= sizes.get(name, 0)
    current = peak = 0
    for change in alive:
        current += change
        peak = max(peak, current)
    return round(weights / MB, 2), round(peak / MB, 2)

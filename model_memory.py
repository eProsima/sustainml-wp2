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
"""Memory footprint of a Hugging Face model, computed from its config without its weights.

It runs in a fresh process (config_memory_mb): the UPMEM simulator used by the HW node replaces
PyTorch's layers for the whole process, and the memory must be that of PyTorch's own layers.
"""

import multiprocessing
import queue as queue_module
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

MB = 1024 * 1024


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


# The model built from its config without weights (on the meta device), as the HW node loads it:
# float16
def _model_without_weights(model_name, hf_token):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    with init_empty_weights(include_buffers=True):
        if getattr(config, "is_encoder_decoder", False):
            model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
        else:
            try:
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
            except Exception:
                model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(torch.float16).eval()
    model.tie_weights()
    return model, config


# Weights (MB): float16 parameters (tied weights counted once) plus buffers
def _weights_mb(model):
    params = sum(p.numel() for p in model.parameters()) * 2
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
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
        results.put((_weights_mb(model), _working_memory_mb(model, config)))
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

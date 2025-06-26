# Copyright 2023 SustainML Consortium
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
"""SustainML HW Resources Provider Node Implementation."""

from sustainml_py.nodes.HardwareResourcesNode import HardwareResourcesNode
from rptu_framework import integration as rptu_integration

# Managing UPMEMEM LLM
import upmem_llm_framework as upmem_layers
import transformers
import onnxruntime
import os
import signal
import threading
import time
import json
import torch
import yaml

# Whether to go on spinning or interrupt
running = False

# ONNX Model-based testing class
class ONNXModel(torch.nn.Module):
    def __init__(self, onnx_model_path):
        super(ONNXModel, self).__init__()
        self.onnx_session = onnxruntime.InferenceSession(onnx_model_path)

    def forward(self, inputs: torch.Tensor):
        input_name = self.onnx_session.get_inputs()[0].name
        np_input = inputs.detach().cpu().numpy()
        outputs = self.onnx_session.run(None, {input_name: np_input})

        if len(outputs) == 1:
            return torch.from_numpy(outputs[0])

        elif len(outputs) == 2:
            bounding_boxes = torch.from_numpy(outputs[0])
            class_scores = torch.from_numpy(outputs[1])
            return bounding_boxes, class_scores

        else:
            return tuple(torch.from_numpy(out) for out in outputs)

    # def forward(self, inputs):
    #     # TODO - Make something intelligent to determine the forward method
    #     return torch.nn.functional.softmax(inputs, dim=0)

def load_any_model(model_name, hf_token=None, **kwargs):

    model = None

    try:
        config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"Model configuration loaded: {config}")
        model_class = transformers.AutoModel._model_mapping.get(type(config), None)

        if "llama" in model_class.__name__.lower() or \
        "mistral" in model_class.__name__.lower() or \
        "qwen" in model_class.__name__.lower() or \
        "phi3" in model_class.__name__.lower() or \
        "t5" in model_class.__name__.lower():
            raise ValueError("Models that use 'llama', 'mistral', 'qwen', 'phi3' or 't5' are not supported.")
    except Exception as e:
        raise Exception(f"[ERROR] Could not load model {model_name}: {e}")

    try:
        if model_class is None:
            print(f"No model class found for config type: {type(config)}")
            model = transformers.AutoModel.from_config(config)

        else:
            print(f"Model class found from config: {model_class.__name__}")

            model = model_class(config)
            print(f"Model class from config with config: {model}")
    except Exception as e:
        raise Exception(f"[ERROR] Could not load model {model_name}: {e}")

    if model is None:
        raise Exception(f"Model {model_name} is not currently supported")

    available_token_classes = [
        ("Token", transformers.AutoTokenizer, {}),
        ("Image", transformers.AutoImageProcessor, {"use_fast": True}),
        ("FeatureExtractor", transformers.AutoFeatureExtractor, {}),
        ("Processor", transformers.AutoProcessor, {})
    ]

    for label, token_class, extra_args in available_token_classes:
        try:
            print(f"Try token loaded as {label}")
            tokenizer = token_class.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                **{**extra_args, **kwargs}
            )
            print(f"[OK]")
            break
        except Exception as e:
            print(f"[WARN] Could not load token as {label}: {e}")

    if tokenizer is None:
        raise Exception(f"Error initializing tokenizer for model {model_name}: {e}")

    input = None
    try:
        print(f"Try input created as a {label}")
        # Text
        if label == "Token":
            if tokenizer.eos_token is None:
                tokenizer.eos_token = "<|endoftext|>"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            text = "How to prepare coffee?"
            input = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

        # Image or Video
        elif label == "Image" or label == "FeatureExtractor" or "image" in tokenizer.__class__.__name__.lower():
            from PIL import Image
            import numpy as np

            # Check for video case based on tokenizer class name containing "video"
            if "video" in tokenizer.__class__.__name__.lower():
                # Video case: create a list of 16 frames (all white images)
                arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
                img = Image.fromarray(arr)
                video_frames = [img for _ in range(16)]
                input = tokenizer(
                    images=video_frames,
                    return_tensors="pt",
                )
            else:
                # Image case: create a single white image
                arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
                img = Image.fromarray(arr)
                input = tokenizer(
                    images=img,
                    return_tensors="pt",
                )
            input = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in input.items()}

        # Multimodal
        elif label == "Processor":
            from PIL import Image
            import numpy as np
            # Create a dummy white image
            arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
            img = Image.fromarray(arr)
            text = "How to prepare coffee?"
            # Combine text and image to create input for the processor
            input = tokenizer(text=text, images=img, return_tensors="pt")

        print(f"[OK] Input created correctly as a {label}")

    except Exception as e:
        raise Exception(f"Error creating input for model {model_name}, tokenizer {tokenizer} : {e}")

    return model, tokenizer, input

# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    HardwareResourcesNode.terminate()
    global running
    running = False

# User Callback implementation
# Inputs: ml_model, app_requirements, hw_constraints
# Outputs: node_status, hw
def task_callback(ml_model, app_requirements, hw_constraints, node_status, hw):

    #Variable to store RPTU default model
    rptu_model = os.path.dirname(__file__)+'/rptu_framework/model.onnx'

    latency = 0.0
    power_consumption = 0.0

    global hf_token

    upmem_layers.initialize_profiling_options(simulation=True)
    upmem_layers.profiler_init()

    hw_selected = hw_constraints.hardware_required()[0]

    model_path = ml_model.model_path()
    if isinstance(model_path, (list, tuple)):
        try:
            model_path = ''.join(chr(b) for b in model_path)
        except Exception:
            model_path = ""

    # Use RPTU hw predictor for their devices
    if hw_selected in ["Zynq UltraScale+ ZCU102", "Zynq UltraScale+ ZCU104", "Ultra96-V2", "TySOM-3A-ZU19EG"]:
        print(f"Using ONNX model path")
        try:
            # Use RPTU
            results = rptu_integration.onnx_ml_resource_estimation(rptu_model, hw_selected) # TODO: hw_selected should affect predictor
            print(f"RPTU latency results: {results['Latency']}")
            print(f"RPTU power consumption results: {results['Run_power']}")
            latency = results['Latency']
            power_consumption = results['Run_power']

        except Exception as e:
            print(f"[ERROR] Failed to load/run ONNX at '{model_path}': {e}.")

    # Use UPMEM hw simulator
    else:
        try:
            print(f"Using Hugging Face model")
            hf_token = None
            extra_data_bytes = hw_constraints.extra_data()
            if extra_data_bytes:
                extra_data_str = ''.join(chr(b) for b in extra_data_bytes)
                if extra_data_str:
                    try:
                        extra_data_dict = json.loads(extra_data_str)
                    except json.JSONDecodeError:
                        print(f"[WARN] extra_data no es JSON válido: {extra_data_str!r}")
                        extra_data_dict = {}
                if "hf_token" in extra_data_dict:
                    hf_token = extra_data_dict["hf_token"]
            if hf_token is None:
                raise Exception("HF token was not provided. Please set the HF_TOKEN environment variable.")

            model, tokenizer, input = load_any_model(
                ml_model.model(),
                hf_token=hf_token,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            print(f"Model, Tokenizer and Input loaded successfully")
            print(f"Model: {model}")
            print(f"Tokenizer: {tokenizer}")
            print(f"Input: {input}")

            layer_mapping = {}
            for name, module in model.named_modules():
                if not name:
                    continue
                if len(list(module.children())) == 0:
                    layer_mapping[name.split('.')[-1]] = hw_selected

            raw_last = list(layer_mapping.keys())[-1]
            last_layer = raw_last.split('.')[-1]
            print(f"Last layer for profiling: {last_layer}")    #debug


            print("Mapped leaf modules:")
            for k in (layer_mapping): #debug
                print("  ", k)

            model.eval()  # Put model in evaluation / inference mode

            # noinspection PyUnresolvedReferences
            upmem_layers.profiler_start(
                layer_mapping    = layer_mapping,
                last_layer       = last_layer,
            )
            # In case we want to time the original execution (comment out profiler_start)
            # start = time.time_ns()

            try:
                output = model.generate(
                    **input, do_sample=True, temperature=0.9, min_length=64, max_length=64
                )
            except Exception as e_gen:
                print(f"Error generating output with generate: {e_gen}. Trying forward instead.")
                try:
                    output = model(**input)
                except Exception as e_model:
                    print(f"Error generating output using model: {e_model}")
                    if "decoder_input_ids" not in input and "input_ids" in input:
                        input["decoder_input_ids"] = input["input_ids"]
                    try:
                        output = model(**input)
                    except Exception as e_model2:
                        raise Exception(e_model2)

            # noinspection PyUnresolvedReferences
            upmem_layers.profiler_end()

            latency = upmem_layers.profiler_get_latency()
            power_consumption = upmem_layers.profiler_get_power_consumption()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error testing model on hardware: {e}")
            print(f"Please provide different model")
            hw.hw_description("Error")
            hw.power_consumption(0.0)
            hw.latency(0.0)
            error_message = "Failed to test model on hardware: " + str(e)
            error_info = {"error": error_message}
            encoded_error = json.dumps(error_info).encode("utf-8")
            hw.extra_data(encoded_error)
            return

    hw.hw_description(hw_selected)
    hw.power_consumption(power_consumption)
    hw.latency(latency)
    print(f"Power Consumption: {power_consumption:.8f} W")
    print(f"Latency: {latency} ms")

# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here
    if req.configuration() == "hardwares":
        try:
            res.node_id(req.node_id())
            res.transaction_id(req.transaction_id())

            # Retrieve Hardwares from sim_architectures.yaml
            with open(os.path.dirname(__file__)+'/upmem_llm_framework/sim_architectures.yaml', 'r') as file:
                upmem_devices = yaml.safe_load(file)
            with open(os.path.dirname(__file__)+'/rptu_framework/rptu_devices.yaml', 'r') as file:
                rptu_devices = yaml.safe_load(file)

            # Extract the hardware names
            hardware_names = list(upmem_devices.keys()) + list(rptu_devices.keys())

            if not hardware_names:
                res.success(False)
                res.err_code(1) # 0: No error || 1: Error
            else:
                res.success(True)
                res.err_code(0) # 0: No error || 1: Error
            sorted_architectures = sorted(list(upmem_devices.keys()))
            sorted_rptu_devices = sorted(list(rptu_devices.keys()))
            sorted_hardware_names = ', '.join(sorted_architectures + sorted_rptu_devices)
            print(f"Available Hardwares: {sorted_hardware_names}")
            res.configuration(json.dumps(dict(hardwares=sorted_hardware_names)))

        except Exception as e:
            print(f"Error getting types of hardwares from request: {e}")
            res.success(False)
            res.err_code(1) # 0: No error || 1: Error

    else:
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())
        error_msg = f"Unsupported configuration request: {req.configuration()}"
        res.configuration(json.dumps({"error": error_msg}))
        res.success(False)
        res.err_code(1) # 0: No error || 1: Error
        print(error_msg)

# Main workflow routine
def run():
    node = HardwareResourcesNode(callback=task_callback, service_callback=configuration_callback)
    global running
    running = True
    node.spin()

# Call main in program execution
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    """Python does not process signals async if
    the main thread is blocked (spin()) so, tun
    user work flow in another thread """
    runner = threading.Thread(target=run)
    runner.start()

    while running:
        time.sleep(1)

    runner.join()

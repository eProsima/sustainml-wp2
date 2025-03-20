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

    upmem_layers.initialize_profiling_options(simulation=True)
    upmem_layers.profiler_init()

    layer_mapping = {
        "input_layernorm": "PIM-AI-1chip",
        "q_proj": "PIM-AI-1chip",
        "k_proj": "PIM-AI-1chip",
        "rotary_emb": "PIM-AI-1chip",
        "v_proj": "PIM-AI-1chip",
        "o_proj": "PIM-AI-1chip",
        "output_layernorm": "PIM-AI-1chip",
        "gate_proj": "PIM-AI-1chip",
        "up_proj": "PIM-AI-1chip",
        "down_proj": "PIM-AI-1chip",
        "norm": "PIM-AI-1chip",
        "lm_head": "PIM-AI-1chip",
    }

    # Use model path if available
    if ml_model.model_path() != "":
        print("Using model path")
        # Instantiate the ONNX predictor model
        onnx_model = ONNXModel(ml_model.model_path())
        my_tensor = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        upmem_layers.profiler_start(layer_mapping)
        onnx_model.forward(my_tensor)
        upmem_layers.profiler_end()

    # Use Hugging Face model
    else:
        try:
            hf_token = "token"  # WIP - Please change "token" to your personal token of HF

            model = transformers.AutoModelForCausalLM.from_pretrained(  # Only works with LLM models
                ml_model.model(), token=hf_token
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                ml_model.model(), token=hf_token
            )

            if tokenizer.eos_token is None:
                tokenizer.eos_token = "<|endoftext|>"

            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

            prompt = "How to prepare coffee?"
            inputs = tokenizer(prompt, return_tensors="pt",
                            padding=True, truncation=True,
                            return_token_type_ids=False)

            print(inputs.data["input_ids"][0].shape)

            model.eval()  # Put model in evaluation / inference mode

            print(model)
            print()

            upmem_layers.profiler_start(layer_mapping)
            # In case we want to time the original execution (comment out profiler_start)
            # start = time.time_ns()
            gen_tokens = model.generate(
                inputs.input_ids, do_sample=True, temperature=0.9, min_length=64, max_length=64
            )
            # print ( (time.time_ns() - start)/1e6)
            upmem_layers.profiler_end()

            gen_text = tokenizer.batch_decode(
                gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            print(gen_text)

        except Exception as e:
            print(f"Error testing model on hardware: {e}")
            print(f"Please provide different model")
            return

    hw.hw_description("PIM-AI-1chip")
    hw.power_consumption(upmem_layers.profiler_get_power_consumption())
    hw.latency(upmem_layers.profiler_get_latency())
    print(f"Power Consumption: {upmem_layers.profiler_get_power_consumption():.8f} W")
    print(f"Latency: {upmem_layers.profiler_get_latency()} ms")

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
                architectures = yaml.safe_load(file)

            # Extract the hardware names
            hardware_names = list(architectures.keys())

            if not hardware_names:
                res.success(False)
                res.err_code(1) # 0: No error || 1: Error
            else:
                res.success(True)
                res.err_code(0) # 0: No error || 1: Error
            sorted_hardware_names = ', '.join(sorted(hardware_names))
            print(f"Available Hardwares: {sorted_hardware_names}")
            res.configuration(json.dumps(dict(hardwares=sorted_hardware_names)))

        except Exception as e:
            print(f"Error getting types of hardwares from request: {e}")
            res.success(False)
            res.err_code(1) # 0: No error || 1: Error

    else:
        # Dummy JSON configuration and implementation
        dummy_config = {
            "param1": "value1",
            "param2": "value2",
            "param3": "value3"
        }
        res.configuration(json.dumps(dummy_config))
        res.node_id(req.node_id())
        res.transaction_id(req.transaction_id())
        res.success(True)
        res.err_code(0) # 0: No error || 1: Error

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

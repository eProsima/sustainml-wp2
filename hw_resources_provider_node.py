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
import onnxruntime
import signal
import threading
import time
import torch
import typer
import upmem_llm_framework as upmem_layers

app = typer.Typer(callback=upmem_layers.initialize_profiling_options)
# Whether to go on spinning or interrupt
running = False

# ONNX Model-based testing class
class ONNXModel(torch.nn.Module):
    def __init__(self, onnx_model_path):
        super(ONNXModel, self).__init__()
        self.onnx_session = onnxruntime.InferenceSession(onnx_model_path)

    def forward(self, inputs):
        # TODO - Make something intelligent to determine the forward method
        return torch.nn.functional.softmax(inputs, dim=0)

# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    HardwareResourcesNode.terminate()
    global running
    running = False

# User Callback implementation
# Inputs: ml_model, app_requirements, hw_constraints
# Outputs: node_status, hw
def task_callback(ml_model, app_requirements,  hw_constraints, node_status, hw):

    upmem_layers.profiler_init()

    # Instantiate the ONNX predictor model
    onnx_model = ONNXModel(ml_model.model_path())
    my_tensor = torch.rand(100)

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

    upmem_layers.profiler_start(layer_mapping)
    onnx_model.forward(my_tensor)
    upmem_layers.profiler_end()
    hw.hw_description("PIM-AI-1chip")
    hw.power_consumption(upmem_layers.profiler_get_power_consumption())
    hw.latency(upmem_layers.profiler_get_latency())

# Main workflow routine
def run():
    node = HardwareResourcesNode(callback=task_callback)
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

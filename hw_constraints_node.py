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
"""SustainML HW Resources Provider Node Implementation."""

from sustainml_py.nodes.HardwareConstraintsNode import HardwareConstraintsNode

# Manage signaling
import ctypes
import json
import numpy as np
import signal
import threading
import time

# Whether to go on spinning or interrupt
running = False

# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    HardwareConstraintsNode.terminate()
    global running
    running = False

# User Callback implementation
# Inputs: user_input
# Outputs: node_status, hw_constraints
def task_callback(user_input, node_status, hw_constraints):

    # Default values
    hw_req = "PIM_AI_1chip"
    mem_footprint = 100

    # Check if extra data has been sent and preserve ALL fields (hf_token, model_family, …)
    incoming = {}
    if user_input.extra_data().size() != 0:
        try:
            buffer = ctypes.c_ubyte * user_input.extra_data().size()
            buffer = buffer.from_address(int(user_input.extra_data().get_buffer()))
            extra_data = np.frombuffer(buffer, dtype=np.uint8)
            extra_data_bytes = extra_data.tobytes()
            extra_data_str = extra_data_bytes.decode('utf-8', errors='ignore')
            incoming = json.loads(extra_data_str) if extra_data_str else {}
        except Exception as e:
            print("[HW_CONSTRAINTS] WARN: could not decode extra_data from user_input:", e)
            incoming = {}

    # Pull values (use defaults if missing)
    mem_footprint = int(incoming.get("max_memory_footprint", mem_footprint))
    hw_req = incoming.get("hardware_required", hw_req)
    hf_token = incoming.get("hf_token")
    model_family = incoming.get("model_family")

    # Build outgoing extra_data preserving everything we received
    out_extra = dict(incoming)  # copy
    if hf_token is not None:
        out_extra["hf_token"] = hf_token
    if model_family is not None:
        out_extra["model_family"] = model_family

    # Forward the full extra_data downstream so HW_RESOURCES can read model_family
    try:
        hw_constraints.extra_data(json.dumps(out_extra).encode("utf-8"))
    except Exception as e:
        print("[HW_CONSTRAINTS] WARN: cannot set hw_constraints.extra_data:", e)

    hw_constraints.max_memory_footprint(mem_footprint)
    hw_constraints.hardware_required([hw_req])

# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here

    # Case not supported
    res.node_id(req.node_id())
    res.transaction_id(req.transaction_id())
    error_msg = f"Unsupported configuration request: {req.configuration()}"
    res.configuration(json.dumps({"error": error_msg}))
    res.success(False)
    res.err_code(1) # 0: No error || 1: Error
    print(error_msg)


# Main workflow routine
def run():
    node = HardwareConstraintsNode(callback=task_callback, service_callback=configuration_callback)
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

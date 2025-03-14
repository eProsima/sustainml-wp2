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

    # Check if extra data has been sent
    if user_input.extra_data().size() != 0:
        buffer = ctypes.c_ubyte * user_input.extra_data().size()
        buffer = buffer.from_address(int(user_input.extra_data().get_buffer()))
        extra_data = np.frombuffer(buffer, dtype=np.uint8)
        extra_data_str = extra_data.tobytes().decode('utf-8', errors='ignore')
        try:
            json_obj = json.loads(extra_data_str)
            if json_obj is not None:
                mem_footprint = int(json_obj["max_memory_footprint"])
                hw_req = json_obj["hardware_required"]
        except:
            print("Extra data is not a valid JSON object, using default values")

    # TODO parse other possible data hidden in the extra_data field, if any
    # TODO populate the hw_constraints object with the required data

    hw_constraints.max_memory_footprint(mem_footprint)
    hw_constraints.hardware_required([hw_req])

# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here

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

# upmem_llm_framework (vendored)

Copy of the `src/upmem_llm_framework` package from https://github.com/eProsima/upmem_llm_framework,
a fork of https://github.com/upmem/upmem_llm_framework, based on commit `f4ed113`.

Licensed under the MIT License, Copyright (c) 2024 UPMEM. See `LICENSE`.

## Origin of each file

Code identical to the fork (only the license header comment may differ):

- `architectures_schema.json`
- `base_architecture.py`
- `options.py`
- `sim_architectures.py`
- `sim_architectures.yaml`
- `simulator.py`
- `utils.py`

Modified by eProsima for SustainML:

- `__init__.py`: also imports `upmem_llm_framework.options` and exports `profiler_get_power_consumption` and `profiler_get_latency`.
- `profiler.py`: adds `get_latency()` and `get_power_consumption()`, guards the profiler summary against division by zero, and renames the summary banner to `HW PROFILER OUTPUT`.
- `pytorch_upmem_layers.py`: adds `profiler_get_power_consumption()` and `profiler_get_latency()`, passes `layer_obj` to `profiler.forward_end` in the `UPM_Conv1D` layer wrapper, and removes a debug print.

In every file that carries a UPMEM header, the original copyright line is kept and a license line
pointing to the MIT License, under which the project is published, has been added. In files where it was present,
this line replaces the original confidentiality notice.

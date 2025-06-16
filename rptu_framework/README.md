# Generic Generator
This project aims to be a generic generator for HLS projects. It is based on the FINN-hls library. The goal it to take any  model in onnx format and generate a HLS project that can be used to run the model on a FPGA. The project is still in development and is not yet ready for production use.

One of the packages 'pydot' may require installation: 

```console
sudo apt-get install graphviz
```

- `dnn_config.hpp` : The header file that contains the configuration of the model (kernel size, number of neurons, number of channels, etc.)
- `dnn_params.hpp` : The header file that contains the weights and activations of the model


## The software library
- `node_hls.py`: This file contains a definition of a class used to operate on a node of a graph (layer of a model)
- `network_hls.py`: This file contains a definition of a class used to operate on a complete graph (a complete model)

## Avalible scripts
- `onnx_to_hls.py`: This script takes an onnx model as an input and generates the configuration files, images of the model, ...

```console
python3 onnx_to_hls.py --model models/q_mbf_relu6_s1_glint360k.onnx 
```

## Hardware implementation
- `facenet_new_hls` - contains the hardware library and the software library
- Build a shared .os library based on the HLS implementation of the corresponding PyTorch model:

```console
cd facenet_new_hls/build
cmake ..
make
```

- `facenet_new_hls/demo/inference.py`: This script takes a config file as an input and reads PyTorch model to check accuracy and generate onnx file

```console
python3 inference.py -config config/q_mbf_relu6_s1_glint360k.py
python3 inference.py -config config/brevitas_mbf_in8_w8_relu8_glint360k.py
```

- `facenet_new_hls/demo/inference.py` - has a `hls_inference_wrapper` function, which is a python wrapper for a shared .os library that is built based on the HLS implementation of the corresponding PyTorch model

## How to add a new layer?
To do so you need to follow the steps below:
1. In `network_hls.py:onn_to_networkx()` define how the generator should extract the node information from the onnx model
2. In `node_hls.py` modify the `__init__()` function to include the new layer
3. In `node_hls.py` add two functions one under `get_macros()` to generate the mactros for the layer and one under `get_instance()` to generate the layer definition, if the function has params consider also modifying the `get_params()` function

## ToDos:
- [ ] `generate_config_hpp` function in `network_hls.py` needs updates to handel multiple top levels
- [ ] we need to check if all parts of the code consider op_type as a list or as a string
- [ ] `merge_several_nodes` function in `network_hls.py` needs to be updated to probably handel int quantization, so it would correctly merge M0 and zero point
- [ ] The case of a split before the first layer in the model needs to be handled
- [ ] The names of the first and last layer in the model should be fixed to avoid modifying the tb with every new model
- [ ] `ConcatStreams_Batch` should be modifed to allow different number of channels in the input streams
- [ ] Currently bit widthes are hardcoded in the code in `update_nodes_default` for onnx models we could extract that from the onnx file.
- [ ] Handel the case of parallel layers running at different speeds `network_hls.py:set_fifo_depth()`
- [X] `--input_mem_width` and `--output_mem_width` are overwriten somewhere in the code and are not used

## notes:
- `M0` in this implementation is the M value in the int quantization paper from google [here](https://arxiv.org/pdf/1712.05877)
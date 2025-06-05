"""
    This file containes functions used to export a torch model to onnx and generate with it a json file containing the information needed to quantize the model.
"""
import torch
import onnx
import json
import torch.nn as nn
from .quantization import QBatchNorm2d, QConv2d, QLinear, QReLU6


def replace_layers(quant_module):
    """
        Takes a quantized model and replaces all its quantized layers with their corresponding non-quantized layers.
        This is done to allow the model to be exported to onnx. Conv and linear layers are initialized with the quantized weights and biases.
        Args:
            quant_module: A quantized model.
        Returns:
            A float version of the model.
    """
    for name, module in quant_module.named_children():
        if isinstance(module, QConv2d):
            tmp = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None, module.padding_mode)
            tmp.weight = nn.Parameter(module.quantize_weight(module.weight))
            tmp.bias = nn.Parameter(module.quantize_bias(module.bias))
            quant_module.add_module(name, tmp)
        elif isinstance(module, QLinear):
            tmp = nn.Linear(module.in_features, module.out_features, module.bias is not None)
            tmp.weight = nn.Parameter(module.quantize_weight(module.weight))
            tmp.bias = nn.Parameter(module.quantize_bias(module.bias))
            quant_module.add_module(name, tmp)
        elif isinstance(module, (QBatchNorm2d, torch.nn.modules.batchnorm.BatchNorm2d)):
            tmp = nn.BatchNorm2d(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
            tmp.weight = module.weight
            tmp.bias = module.bias
            tmp.running_mean = module.running_mean
            tmp.running_var = module.running_var
            quant_module.add_module(name, tmp)
        elif isinstance(module, QReLU6):
            quant_module.add_module(name, nn.ReLU6(module.inplace))
        elif isinstance(module, nn.Sequential):
            quant_module.add_module(name, replace_layers(module))
    return quant_module


def quantizer_to_dict(quantizer):
    """
        Takes a quantizer and returns a dictionary containing the quantization information.
        This is done based on the quantizer's string representation.
        Args:
            quantizer: A quantizer.
        Returns:
            A dictionary containing the quantization information.
    """
    quantizer_dict = {}
    for i in str(quantizer).split(','):
        quantizer_dict[i.split(':')[0].strip()] = i.split(':')[1].strip()
    return quantizer_dict


def extract_quant_info(quant_module, prefix='/'):
    """
        Takes a quantized model and returns a dictionary containing the quantization information.
        Args:
            quant_module: A quantized model.
            prefix: A string used to prefix the name of the layers, this is used to generate the same name as the one used in the onnx model.
        Returns:
            A dictionary containing the quantization information.
    """
    quant_info = {}
    onnx_name = {'QConv2d': 'Conv', 'QLinear': 'Gemm', 'QBatchNorm2d': 'BatchNormalization', 'QReLU6': 'Clip'}
    for name, module in quant_module.named_children():
        if isinstance(module, nn.Sequential):
            quant_info.update(extract_quant_info(module, prefix+name+prefix+name+'.'))
        else:
            layer_quant = {}
            if isinstance(module, (QConv2d, QLinear)):
                layer_quant['weight'] = quantizer_to_dict(module.quantize_weight)
                layer_quant['bias'] = quantizer_to_dict(module.quantize_bias)
                layer_quant['in_activation'] = quantizer_to_dict(module.quantize_in_activation)
                layer_quant['out_activation'] = quantizer_to_dict(module.quantize_out_activation)
            elif isinstance(module, QReLU6):
                layer_quant['quantize_out_activations'] = quantizer_to_dict(module.quantize_out_activations)
            if hasattr(module, 'profiler'):
                for p in module.profiler:
                    layer_quant[p.name+'_profiler'] = {'min': p.minimum.item(), 'max': p.maximum.item(), 'int_bit_width': p.return_int_bit_width()}
            if layer_quant:
                name = name+'/'+onnx_name[type(module).__name__]
                quant_info[prefix+name] = layer_quant
    return quant_info


def dict_to_json_file(quant_info, json_file):
    """
        Takes a dictionary and saves it to a json file.
        Args:
            quant_info: A dictionary containing the quantization information.
            json_file: The path of the json file to save the dictionary to.
    """
    with open(json_file, 'w') as f:
        json.dump(quant_info, f, indent=4)


def export_to_onnx_json(model, export_path, input_size, verbose=False,  do_constant_folding=False, training=torch.onnx.TrainingMode.TRAINING):
    """
        Takes a quatnized model, extractes it's quantization information and saves it to a json file.
        Then replaces all the quantized layers with their corresponding non-quantized layers and exports the float model to onnx.
        Args:
            model: A quantized model.
            export_path: The path to save the onnx model to.
    """
    quant_info = extract_quant_info(model)
    dict_to_json_file(quant_info, export_path+'.json')
    print(f'Quantization information saved to {export_path}.json')
    converted_model = replace_layers(model)
    torch.onnx.export(converted_model.cuda(), torch.randn(input_size).cuda(), export_path+'.onnx', verbose=verbose, do_constant_folding=do_constant_folding, training=training)
    onnx_model = onnx.load(export_path+'.onnx')
    onnx.checker.check_model(onnx_model)
    print(f'Onnx model saved to {export_path}.onnx')

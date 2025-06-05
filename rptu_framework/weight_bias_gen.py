'''
    This file contains functions responsible for generating the weight and bias strings for each layer
'''

import torch
import math
from onnx import numpy_helper
from .utils import tensor_to_stream, onnx_to_stream


def conv_weight_bias(layer_info, weight, wdtype, bias, bdtype, file_object, finn_structure=False):
    '''This function generate the weight and bias strings for conv & transposed conv layers
    Args:
        layer_info: list that contains [layer_name, kernel, input_channels, output_channels, simd, pe]
        weight: weight object from onnx_model.graph.initializer
        wdtype: data type of weights, can be int, float or fixed
        bias: bias object from onnx_model.graph.initializer
        bdtype: data type of bias, can be int, float or fixed
        file_object: file object to write the finn weight and bias to
        finn_structure: if true, the finn structure will be used, otherwise the normal arrays will be used
    '''
    if isinstance(weight, torch.Tensor):
        weights_gen = tensor_to_stream(weight)
    else:
        weights_gen = onnx_to_stream(weight)
    layer_name = layer_info[0]
    simd = layer_info[4]
    pe = layer_info[5]
    assert wdtype in ['int', 'float', 'fixed'], f'invalid data type {wdtype}'
    if finn_structure:
        file_object.write(f'FixedPointWeights<{layer_name.upper()}SIMD, {layer_name.lower()}weight_dtype,{layer_name.upper()}PE,{layer_name.upper()}WMEM> {layer_name.lower()}weights =\n{{{{\n')
    else:
        file_object.write(f'{layer_name.lower()}weight_dtype {layer_name.lower()}weights[{layer_name.upper()}PE][{layer_name.upper()}WMEM][{layer_name.upper()}SIMD] =\n{{\n')
    tiles = math.prod(weight.shape)//(simd*pe)
    if wdtype == 'int':
        for _ in range(pe*tiles*simd-1):
            file_object.write(f'{int(next(weights_gen))}, ')
        file_object.write(f'{int(next(weights_gen))}}};\n\n')
    elif wdtype == 'float':
        for _ in range(pe*tiles*simd-1):
            file_object.write(f'{next(weights_gen)}, ')
        file_object.write(f'{next(weights_gen)}}};\n\n')
    else:
        raise NotImplementedError('fixed point weights not supported yet')
        # weight_bit_width = int(layer.quant_weight_bit_width())
        # for p in range(pe):
        #     file_object.write('{\n')
        #     for t in range(tiles):
        #         val=''
        #         for s in range(simd):
        #             val=f'{val}{bin_digits(next(weights_gen),weight_bit_width)}'
        #         file_object.write(f'"0x{val}", ' if t!=tiles-1 else f'"0x{val}"')
        #     file_object.write('},\n' if p!=pe-1 else '}\n')
        # file_object.write('}};\n\n')
    try:
        next(weights_gen)  # Making sure all weights were consumed
        raise ValueError(f'weights_gen has left over data at layer: {layer_name}')
    except StopIteration:
        pass

    if bias is not None:
        tiles = layer_info[3]//pe
        bias_iter = iter(bias)
        if finn_structure:
            raise NotImplementedError('finn_structure biases not supported yet')
            # output_bit_width = int(layer.quant_output_bit_width()) if isinstance(layer,nn.Conv2d) else layer_info[-1]
            # act_max_value = 2**(output_bit_width-1)-1
            # file_object.write(f'BiasActivation<{layer_name.upper()}BMEM,{layer_name.upper()}PE, {layer_name.lower()}output_dtype,data_t,{act_max_value}> {layer_name.lower()}biases =\n{{{{\n')
        else:
            file_object.write(f'{layer_name.lower()}bias_dtype {layer_name.lower()}biases[{layer_name.upper()}PE][{layer_name.upper()}BMEM] =\n{{\n')
        if wdtype == 'int':
            for _ in range(pe*tiles*simd-1):
                file_object.write(f'{int(next(bias_iter))}, ')
            file_object.write(f'{int(next(bias_iter))}}};\n')
        elif bdtype == 'float':
            for _ in range(pe*tiles-1):
                file_object.write(f'{float(next(bias_iter))}, ')
            file_object.write(f'{float(next(bias_iter))}')
            file_object.write('\n};\n')
        else:
            raise NotImplementedError('fixed point biases not supported yet')
            # input_scale = layer_info[3]
            # output_scale = layer_info[4]
            # multiplier=(input_scale*layer.quant_weight_scale())/output_scale
            # multiplier=multiplier.squeeze()
            # multiplier_iter=iter(multiplier)
            # for p in range(pe):
            #     file_object.write('{\n')
            #     for t in range(tiles-1):
            #         file_object.write(f'{{{int(next(bias_iter))},{next(multiplier_iter).item()}}}, ')
            #     file_object.write(f'{{{int(next(bias_iter))},{next(multiplier_iter).item()}}}')
            #     file_object.write('},\n' if p!=pe-1 else '}')
            # file_object.write('\n};\n')
        try:
            next(bias_iter)
            raise ValueError(f'bias_iter has left over data at layer: {layer_name}')
        except StopIteration:
            pass


def linear_weight_bias(fullyconn_counter, weight, wdtype, bias, bdtype, file_object, finn_structure=False, prefix='FC', weight_name='weights', bias_name='biases'):
    '''This function generate the weight and bias strings for conv & transposed conv layers
    Args:
        fullyconn_counter: Layer counter
        weight: weight object from onnx_model.graph.initializer
        wdtype: data type of weights, can be int, float or fixed
        bias: bias object from onnx_model.graph.initializer
        bdtype: data type of bias, can be int, float or fixed
        file_object: file object to write the finn weight and bias to
        finn_structure: if true, the finn structure will be used, otherwise the normal arrays will be used
    '''
    layer_name = f'{prefix}_{fullyconn_counter}'
    if finn_structure:
        file_object.write(f'FixedPointWeights<{layer_name}_SIMD, {layer_name.lower()}_weight_dtype,{layer_name}_PE,{layer_name}_WMEM> {layer_name.lower()}_{weight_name} =\n')
    else:
        file_object.write(f'{layer_name.lower()}_weight_dtype {layer_name.lower()}_{weight_name} [{layer_name}_PE][{layer_name}_WMEM][{layer_name}_SIMD] =\n')
    file_object.write('{\n')
    if wdtype == 'int':
        raise NotImplementedError('int weights not supported yet')
    elif wdtype == 'float':
        for row in weight:
            for val in row:
                file_object.write((f'{val}, '))
            file_object.write('\n')
    elif wdtype == 'fixed':
        raise NotImplementedError('fixed weights not supported yet')
    file_object.write('\n};\n')
    if bias is not None:
        if finn_structure:
            file_object.write(f'BiasActivation<{layer_name}_BMEM,{layer_name}_PE, {layer_name.lower()}_activation_dtype,{layer_name.lower()}_bias_dtype> {layer_name.lower()}_{bias_name} =\n')
        else:
            file_object.write(f'{layer_name.lower()}_bias_dtype {layer_name.lower()}_{bias_name} [{layer_name}_PE][{layer_name}_BMEM] =')
        file_object.write('{{\n')
        if bdtype == 'int':
            raise NotImplementedError('int bias not supported yet')
        elif bdtype == 'float':
            for val in bias:
                file_object.write((f'{val}, '))
        elif bdtype == 'fixed':
            raise NotImplementedError('fixed bias not supported yet')
        file_object.write('\n}};\n\n')

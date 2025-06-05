import math

import numpy as np
from .utils import calc_bit_width, profile_array
from .defaults import SupportedNodes, DefaultValues


def get_brevitas_info(node_idx, nx_graph, initilizers_dict, input_var=True):
    node = nx_graph.nodes[node_idx]
    if input_var:
        if node['op_type'] == SupportedNodes.dequantize:
            if node['var_name'] in initilizers_dict:  # if the variabel is stored in the correct node
                return {'init_name': node['var_name'], 'scale': float(node['scale']), 'zero_point': int(node['zero_point'])}
            else:
                source_node_idx = next(nx_graph.predecessors(node_idx))
                source_node = nx_graph.nodes[source_node_idx]
                if source_node['op_type'] == SupportedNodes.quantize:
                    assert node['scale'] == source_node['scale'] and node['zero_point'] == source_node['zero_point']
                    return {'init_name': source_node['var_name'], 'scale': float(source_node['scale']), 'zero_point': int(source_node['zero_point'])}
                elif source_node['op_type'] == SupportedNodes.clip:
                    source_source_node = nx_graph.nodes[next(nx_graph.predecessors(source_node_idx))]
                    assert source_source_node['op_type'] == SupportedNodes.quantize
                    assert node['scale'] == source_source_node['scale'] and node['zero_point'] == source_source_node['zero_point']
                    return {'init_name': source_source_node['var_name'], 'scale': float(source_source_node['scale']), 'zero_point': int(source_source_node['zero_point']), 'range': [int(source_node['min']), int(source_node['max'])]}
                else:
                    raise NotImplementedError(f'Input should start with dequantize -> (Quantize or clip), got {source_node["op_type"]}')
        elif node['op_type'] in [SupportedNodes.maxpool, SupportedNodes.concat]:  # Usually the output of the maxpool is not quantized, so we need to go back to it's input to get the quantization info
            return get_brevitas_info(next(nx_graph.predecessors(node_idx)), nx_graph, initilizers_dict)
    else:
        if node['op_type'] == SupportedNodes.quantize:
            next_node = nx_graph.nodes[next(nx_graph.successors(node_idx))]
            if next_node['op_type'] == SupportedNodes.dequantize:
                assert node['scale'] == next_node['scale'] and node['zero_point'] == next_node['zero_point']
                return {'scale': float(next_node['scale']), 'zero_point': int(next_node['zero_point'])}


def merge_brevitas(nx_graph, initilizers_dict):
    for i in nx_graph.nodes():
        node = nx_graph.nodes[i]
        if node['op_type'] in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear, SupportedNodes.relu]:
            node_predecessors = list(nx_graph.predecessors(i))
            node_successors = list(nx_graph.successors(i))
            input_quant = get_brevitas_info(node_predecessors[0], nx_graph, initilizers_dict)
            if input_quant is not None:
                if 'range' in input_quant:
                    node['in_bits'] = calc_bit_width(input_quant['range'])
                else:
                    node['in_bits'] = DefaultValues.brevitas_default_act_bits
                node['in_int_bits'] = int(node['in_bits'] + math.log2(input_quant['scale']))
            if len(node_successors) > 0:
                output_quant = get_brevitas_info(node_successors[0], nx_graph, initilizers_dict, input_var=False)
                if output_quant is not None:
                    if 'range' in output_quant:
                        node['out_bits'] = calc_bit_width(output_quant['range'])
                    else:
                        node['out_bits'] = DefaultValues.brevitas_default_act_bits
                    node['out_int_bits'] = int(node['out_bits'] + math.log2(output_quant['scale']))
            if len(node_predecessors) > 1:
                if nx_graph.nodes[node_predecessors[1]]['op_type'] == SupportedNodes.transpose:
                    node_predecessors[1] = list(nx_graph.predecessors(node_predecessors[1]))[0]
                weight_quant = get_brevitas_info(node_predecessors[1], nx_graph, initilizers_dict)
                if weight_quant is not None:
                    arr_range = weight_quant.get('range', profile_array(initilizers_dict[weight_quant['init_name']]))
                    node['weight_bits'] = calc_bit_width(arr_range)
                    node['weight_int_bits'] = int(node['weight_bits'] + math.log2(weight_quant['scale']))
                    node['weight'] = np.copy(initilizers_dict[weight_quant['init_name']])
                    node['weight'] = quantize_linear(node['weight'], weight_quant['scale'], weight_quant['zero_point'], arr_range)
                    node['weight'] = dequantize_linear(node['weight'], weight_quant['scale'], weight_quant['zero_point'])
            if len(node_predecessors) > 2:
                bias_quant = get_brevitas_info(node_predecessors[2], nx_graph, initilizers_dict)
                if bias_quant is not None:
                    arr_range = bias_quant.get('range', profile_array(initilizers_dict[bias_quant['init_name']]))
                    node['bias_bits'] = calc_bit_width(arr_range)
                    node['bias_int_bits'] = int(node['bias_bits'] + math.log2(bias_quant['scale']))
                    node['bias'] = np.copy(initilizers_dict[bias_quant['init_name']])
                    node['bias'] = dequantize_linear(node['bias'], bias_quant['scale'], bias_quant['zero_point'])
            if node['op_type'] == SupportedNodes.linear:
                node['input_shape'].append(node['weight'].shape[1])
        elif node['op_type'] in [SupportedNodes.maxpool, SupportedNodes.concat]:
            input_quant = get_brevitas_info(node_predecessors[0], nx_graph, initilizers_dict)
            if input_quant is not None:
                node['in_bits'] = DefaultValues.brevitas_default_act_bits
                node['in_int_bits'] = int(node['in_bits'] + math.log2(input_quant['scale']))
            node['out_bits'] = node['in_bits']
            node['out_int_bits'] = node['in_int_bits']
    return nx_graph


def quantize_linear(array, scale, zero_point, range):
    # The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
    return np.clip(array / scale + zero_point, range[0], range[1])


def dequantize_linear(quantized, scale, zero_point):
    # The dequantization formula is y = (x - x_zero_point) * x_scale
    return (quantized - zero_point) * scale

'''
    This file contains utility functions
'''


import brevitas
import brevitas.quant_tensor
from onnx import numpy_helper
from torchinfo import summary
import math
import numpy as np

import errno
import os
from .defaults import DefaultValues

def shift_bit_length(x):
    return 1<<(x-1).bit_length()


def check_path(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def save_to_file(data, path):
    check_path(os.path.dirname(path))
    with open(path, 'w') as f:
        print(f"Save on disc to {path}")
        f.write(data)


def get_weight_bias(initializers, w_name, b_name):
    """Extracts weights and biases from the model
       Loop over the list of initilizers looking for weight and bias of a specific layer based on the name.
    Args:
        initializers: List of initializers
        w_name: Name of the weights tensor
        b_name: Name of the biases tensor
    Returns:
        A tuple containing the weights and biases
    """
    w = None
    b = None
    for i in range(0, len(initializers)):
        if initializers[i].name == w_name:
            w = numpy_helper.to_array(initializers[i])
        if initializers[i].name == b_name:
            b = numpy_helper.to_array(initializers[i])
        if w is not None and b is not None:
            return w, b
    raise ValueError('Weights or biases not found')


def bin_digits(val, num_bits):
    '''Returns the correct binary value of val

    Args:
        val: Decimal value to convert to binary
        num_bits: Number of bits to use
    Returns:
        A string containing the resulting binary value without 0b or minus
    Thanks Jonas:)
    '''
    if num_bits % 4 != 0:
        num_bits = (num_bits+4)-(num_bits % 4)
    s = bin(val & int('1' * num_bits, 2))[2:]
    d = f'{s:0>{num_bits}}'
    res = ''
    for i in range(0, len(d), 4):
        res = f'{res}{hex(int(d[i:i+4],base=2))[2:]}'
    return res


def onnx_to_stream(array):
    '''Takes in a numpy array and yields it's values in depth wise raster order
        (i.e. All channels of pixel (0,0) first, then all channels of pixel (0,1), etc.)

    Args:
        array: A numpy array to be converted to a stream must be 3D or 4D
    Yields:
        Values in the array in depth wise raster order
    '''
    assert array.ndim in [3, 4], f'Only 3D and 4D tensors are supported, got {array.ndim}D'
    if array.ndim == 3:
        array = array.unsqueeze(0)
    for b in range(array.shape[0]):
        for r in range(array.shape[2]):
            for c in range(array.shape[3]):
                for ch in range(array.shape[1]):
                    yield array[b][ch][r][c]


def tensor_to_stream(tensor):
    '''Takes in torch tensor or quant tensor and yields it's values in depth wise raster order
        (i.e. All channels of pixel (0,0) first, then all channels of pixel (0,1), etc.)

    Args:
        tensor: A torch tensor or quant tensor to be converted to a stream must be 3D or 4D
    Yields:
        Values in the tensor in depth wise raster order
    '''
    if isinstance(tensor, brevitas.quant_tensor.QuantTensor):
        tensor = tensor.value
    assert tensor.dim() in [3, 4], f'Only 3D and 4D tensors are supported, got {tensor.dim()}D'
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    for b in range(tensor.shape[0]):
        for r in range(tensor.shape[2]):
            for c in range(tensor.shape[3]):
                for ch in range(tensor.shape[1]):
                    yield tensor[b][ch][r][c]


def tensor_to_stream_tr_conv(tensor):
    '''Same as the original tensor_to_stream, but torch.nn.ConvTranspose2d uses a different weight order compared to torch.nn.Conv2d

    Args:
        tensor: A torch tensor or quant tensor to be converted to a stream must be 3D or 4D
    Yields:
        Values in the tensor in depth wise raster order
    '''

    if isinstance(tensor, brevitas.quant_tensor.QuantTensor):
        tensor = tensor.value
    assert tensor.dim() == 4, f'Only 3D and 4D tensors are supported, got {tensor.dim()}D'
    for a in range(tensor.shape[2]):
        for b in range(tensor.shape[3]):
            for c in range(tensor.shape[1]):
                for d in range(tensor.shape[0]):
                    yield tensor[d][c][a][b]


def rcc_to_dict(rcc):
    '''This function takes a google.protobuf.pyext._message.RepeatedCompositeContainer
    object and converts it to a dictionary.
    Args:
        rcc: a google.protobuf.pyext._message.RepeatedCompositeContainer object
    Returns:
        a dictionary containing the rcc's elements
    '''
    res = {}
    for obj in rcc:
        if obj.type == 'INTS' or obj.type == 7:
            res[obj.name] = obj.ints
        elif obj.type == 'INT' or obj.type == 2:
            res[obj.name] = obj.i
        elif obj.type == 'FLOAT' or obj.type == 1:
            res[obj.name] = obj.f
        elif obj.type == 'STRING' or obj.type == 3:
            res[obj.name] = obj.s
        elif obj.type == 'TENSOR' or obj.type == 4:
            res[obj.name] = obj.t
        else:
            raise ValueError(f'Unknown type {obj.type}, name {obj.name}')
    return res


def value_info_dims(v):
    '''This function takes a ValueInfo object and returns its dimensions.
    Args:
        v: a ValueInfo object Like This:
        name: "/features/features.0/Conv_output_0"
        type {
        tensor_type {
            elem_type: 1
            shape {
            dim {
                dim_value: 1
            }
            dim {
                dim_value: 64
            }
            dim {
                dim_value: 224
            }
            dim {
                dim_value: 224
            }
            }
        }
        }
    Returns:
        a list containing the dimensions of the ValueInfo object v like this: [1, 64, 224, 224]

    '''
    return [v.type.tensor_type.shape.dim[i].dim_value for i in range(len(v.type.tensor_type.shape.dim))]


def model_summarizer(model, input_shape):
    ''' Takes a model object & uses torchinfo summary to generate a summary list & then loop over it to exclude unnecessary layers
    Args: model - a pytorch model object
    Returns: a list containing the summary of the model
    '''
    summary_list = summary(model, input_shape, verbose=0, depth=2).summary_list
    summarized = []
    for l in summary_list:
        if l.class_name in ['Conv2d', 'QuantConv2d', 'QuantMaxPool2d', 'ConvTranspose2d', 'QuantConvTranspose2d', 'UNet_down_block', 'Quant_unet_down_block', 'UNet_up_block', 'Quant_unet_up_block', 'Sequential', 'QuantIdentity', 'ReLU', 'QuantReLU', 'ReLU6', 'BatchNorm2d']:
            summarized.append(l)
    return summarized


def parse_model_init(key_value_pairs):
    """ Generated by ChatGPT :) modified by me :)
    Parses a list of key-value pairs separated by colons (:)
    and returns a dictionary with the values cast to their original type.

    Args:
        key_value_pairs (str): A list of key-value pairs separated by colons (:).

    Returns:
        dict: A dictionary with the key-value pairs from the input string, where the values
        are casted to their original type (int, bool, or str).

    Example:
        key_value_pairs = "name:John age:25 married:t"
        output_dict = parse_model_init(key_value_pairs)
        print(output_dict)
        # Output: {'name': 'John', 'age': 25, 'married': True}
    """
    result = {}
    for item in key_value_pairs:
        key, value = item.split(':')
        if value.isdigit():
            result[key] = int(value)
        elif value.lower() in ['true', 't', '1']:
            result[key] = True
        elif value.lower() in ['false', 'f', '0']:
            result[key] = False
        else:
            result[key] = value
    return result

def get_divisors_under_limit(mem_width, factor, channels):
    '''
        This function takes a memory width and a factor and returns the divisors of the memory width that are less than or equal to the factor.
        This is needed to find PE and SIMD values for the first and last node in the graph.
    '''
    divisors = [i for i in range(1, channels + 1) if mem_width % i == 0 and i <= factor and channels % i == 0]
    return max(divisors) if divisors else 1

def find_simd_pe(inch, outch, factor, constant=None, edge_simd_pe=None, interface_width=None):
    if constant == 'pe':
        for simd in range(factor, inch+1, 1):
            if inch % simd == 0:
                return simd, 1
    elif constant == 'simd':
        for pe in range(factor, outch+1, 1):
            if outch % pe == 0:
                return 1, pe
    if edge_simd_pe == 'simd':
        simd = get_divisors_under_limit(interface_width, factor, inch)
        if simd < factor:
            for pe in range(factor, outch+1, 1):
                if outch % pe == 0:
                    return simd, pe
    elif edge_simd_pe == 'pe':
        pe = get_divisors_under_limit(interface_width, factor, outch)
        if pe < factor:
            for simd in range(factor, inch+1, 1):
                if inch % simd == 0:
                    return simd, pe
    square_root = math.ceil(math.sqrt(factor))
    if inch % square_root == 0 and outch % square_root == 0:
        if square_root <= inch and square_root <= outch:
            return square_root, square_root
        else:
            return inch, outch
    for simd in range(square_root, inch+1, 1):
        if inch % simd == 0:
            for pe in range(square_root, outch+1, 1):
                if outch % pe == 0:
                    if (pe * simd) >= factor:
                        return simd, pe
    return inch, outch


def equalize_simd_pe(nx_graph, node, source_node=None, direction=None):
    '''
        A recursive function that check if the node's successors have the same SIMD as the PE of the node, and if the node's predecessors have the same PE as the SIMD of the node.
        If not, it changes the values to the maximum of the two. Retuning it to the previous call so it can also be set there.
    '''
    print(f'Equalizing SIMD and PE for node {node.name}...')
    if direction in ['pe', 'both']:
        node_successors = list(nx_graph.successors(node))
        if source_node in node_successors:
            node_successors.remove(source_node)
        print([x.name for x in node_successors])
        if len(node_successors) > 0:
            max_value = max([n.simd for n in node_successors]+[node.pe])
            node.pe = max_value
            for succ in node_successors:
                succ.simd = max_value
                equalize_simd_pe(nx_graph, succ, node, 'simd')
    if direction in ['simd', 'both']:
        node_predecessors = list(nx_graph.predecessors(node))
        if source_node in node_predecessors:
            node_predecessors.remove(source_node)
        print([x.name for x in node_predecessors])
        if len(node_predecessors) > 0:
            max_value = max([n.pe for n in node_predecessors]+[node.simd])
            node.simd = max_value
            for pred in node_predecessors:
                pred.pe = max_value
                equalize_simd_pe(nx_graph, pred, node)


def get_source_node(var_name, graph):
    ''' Get the node from the graph by output name
    '''
    for i in graph.node:
        if var_name in i.output:
            return i
    return None


def get_dest_node(var_name, graph):
    ''' Get the node from the graph by input name
    '''
    for i in graph.node:
        if var_name in i.input:
            return i
    return None


def profile_array(array):
    '''
        This function takes an array and returns the minimum and maximum values of the array.
    '''
    return int(np.min(array)), int(np.max(array))


def calc_bit_width(min_max):
    '''
        This function takes a min_max and returns the minimum bit width required to represent the min_max.
        min_max is a tuple of the form (min, max) where min and max are the minimum and maximum values of an integer array.
    '''
    return math.ceil(math.log2(abs(min_max[0])+abs(min_max[1])))


def decompose_float(real_multiplier, num_bits=32):
    '''
        This function takes a real multiplier and returns its integer representation and the number of shifts required to represent it.
        Based on the gemmlowp implementation: https://github.com/google/gemmlowp/blob/16e8662c34917be0065110bfcd9cc27d30f52fdf/doc/quantization_example.cc#L210
        with modifications to support different bit widths.
    '''
    # assert 0.0 < real_multiplier < 1.0, f"real_multiplier must be between 0 and 1, got {real_multiplier}"
    assert num_bits in [16, 32], f"num_bits must be either 16 or 32, got {num_bits}"

    s = 0
    # Bring the real multiplier into the interval [1/2, 1)
    while real_multiplier < 0.5:
        real_multiplier *= 2.0
        s += 1

    # Convert it into a fixed-point number
    q = np.round(real_multiplier * (1 << (num_bits - 1)))
    assert q <= (1 << (num_bits - 1)), f"q must be less than or equal to 2^31, got {q}"

    # Handle the special case when the real multiplier was so close to 1
    if q == (1 << (num_bits - 1)):
        q //= 2
        s -= 1

    assert s >= 0, f"s must be non-negative, got {s}"
    assert q <= np.iinfo(np.int32).max, f"q must be less than or equal to the max int32 value, got {q}"
    if num_bits == 32:
        quantized_multiplier = np.int32(q)
    else:
        quantized_multiplier = np.int16(q)
    right_shift = s

    return quantized_multiplier, right_shift


# def decompose_float(f, bits=31):
#     '''
#         This function takes a float and returns its integer mantissa and exponent.
#         The float is represented as f = mantissa * 2^exponent
#         Args:
#             f: The float to decompose
#             bits: The number of bits to represent the mantissa
#         Returns:
#             A tuple containing the integer mantissa and exponent
#     '''
#     def value_level(value):
#         if value == 0:
#             return (0, 0)
#         # Get the fraction representation of the float
#         mantissa, exponent = math.frexp(value)
#         # Normalize mantissa to be an integer
#         normalized_mantissa = mantissa * (1 << bits)
#         integer_mantissa = int(normalized_mantissa)
#         # Adjust the exponent accordingly
#         exponent -= bits
#         return integer_mantissa, -exponent
#     if isinstance(f, np.ndarray):
#         scales, zero_points = [], []
#         for f_i in f:
#             scale, zero_point = value_level(f_i.item())
#             scales.append(scale)
#             zero_points.append(zero_point)
#         return scales, zero_points
#     else:
#         return value_level(f)


def calc_M0_shift(input_scale, weight_scale, output_scale, num_bits):
    '''
        This function takes the input, weight and output scales and returns an integer value and number of shifts that togeather represents the M value.
        M is calculated as follows:
        M = (input_scale * weight_scale) / output_scale
        Args:
            input_scale: The scale of the input tensor
            weight_scale: The scale of the weight tensor
            output_scale: The scale of the output tensor
            num_bits: The number of bits used to represent the integer value
    '''
    M = (input_scale * weight_scale) / output_scale
    M_int, shift = decompose_float(M, num_bits)
    return M_int, shift


def extract_val_scale_zero_point(node, initilizers_dict):
    ''' Extract the value -This can be weight or bias-, scale and zero point from the dequantizelinear node
    '''
    val = initilizers_dict[node.input[0]]
    scale, zero_point = extract_scale_zero_point(node, initilizers_dict)
    return val, scale, zero_point


def extract_scale_zero_point(node, initilizers_dict):
    ''' Extract the scale and zero point from the dequantizelinear node
    '''
    scale = initilizers_dict[node.input[1]]
    zero_point = initilizers_dict[node.input[2]]
    if isinstance(zero_point, np.ndarray):
        if zero_point.ndim == 0:
            zero_point = zero_point.item()
    return scale, zero_point


def generate_m0_shift_str(m0, shift):
    ''' Generate the string representation of M0 and shift
    '''
    res = ''
    if isinstance(m0, list):
        res += '.m0 = {'
        res += ', '.join([str(x) for x in m0])
        res += '},\n.shift = {'
        res += ', '.join([str(x) for x in shift])
        res += '},\n'
    else:
        res += f'.m0 = {m0}, .shift = {shift},\n'


def calc_bit_width_fixed(value, total_bits=DefaultValues.int_quant_fixed_scale_bits, signed=False):
    """
    Calculates the number of bits required to represent the integer and fractional parts of a value.

    Parameters:
    - value (float): The value to be represented.
    - total_bits (int): The total number of bits available (default is 32).
    - signed (bool): Indicates if the number is signed (default is False).

    Returns:
    - total_bits (int): Total number of bits required to represent the value.
    - int_bits (int): Number of bits assigned to the integer part.
    """
    # Extract integer parts
    if value in [0, 1]:
        return total_bits, 1
    integer_part = int(abs(value))

    # Calculate bits required for the integer part
    int_bits = 0
    if integer_part == 0:
        while int(value) == 0:
            value *= 10
            int_bits -= 1
        int_bits += 1
    else:
        int_bits = math.floor(math.log2(integer_part))+1
        if signed:
            int_bits += 1

    # Ensure the integer bits do not exceed total bits
    int_bits = min(int_bits, total_bits - 1)

    return total_bits, int_bits

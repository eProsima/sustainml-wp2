import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from functools import partial
import math
from .defaults import Dtypes

##############################################################################################################
# Conversion functions
##############################################################################################################

def bindigits(n, bits):
    """
    | Convert integer to twos complement representation.

    :param int n: Integer number to convert
    :param int bits: Number of bits of twos complement
    :return: String representing twos complement
    """
    s = bin(n & int("1" * bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

def float_to_binary_string(q_float, bit_width, int_width):
    """
    | Convert float to binary fixed point representation as needed for HLS ap_fixed datatype.

    :param float q_float: Floating point number to convert
    :param int bit_width: Number of bits of fixed point number
    :param int int_width: Number of integer bits of fixed point number
    :return: String representing binary representation of fixed point number
    """

    frac_width = bit_width - int_width
    mul_fac = 2 ** frac_width
    q_int = int(q_float * mul_fac)
    bin_int = bindigits(q_int, bit_width)
    return str(bin_int)

def conv_bias_mapper_2D(bias, PE, bits, int_bits, dtype=Dtypes.FIXED):
    """
    | Transforms bias of conv layer to a string that can be included in .hpp file.
    | Bias has shape: OUTPUT_CHANNELS.
    | Stores bias in BiasActivation wrapper with ap_fixed type.

    :param bias: Bias of convolutional layer as given by *layer.bias*
    :param PE: Number of parallel computed output channels
    :param str i: index of layer
    :param int bias_bits: Number of bits to use for quantization
    :param int bias_int_bits: Number of integer bits to use for quantization
    :param Dtypes dtype: Type of data to quantize
    :return: String to save in .hpp file
    """
    bias = bias.detach().cpu().numpy()

    output_channels = bias.shape[0]
    bias_string = ""

    base_output_channel_iter = 0
    pe_output_channel_iter = 0

    #bias_string += "{"
    for output_channel in range(output_channels):
        bias_string += "\n"
        fraction_bits = bits - int_bits
        if dtype == Dtypes.INT:
            value = f'{bias[base_output_channel_iter + pe_output_channel_iter]}'
        else:
            value = f'{bias[base_output_channel_iter + pe_output_channel_iter]:.{fraction_bits}}'
        bias_string += value
        #bias_string += str(bias[base_output_channel_iter + pe_output_channel_iter])
        #print("bias_string: ", bias_string)
        pe_output_channel_iter += PE
        if pe_output_channel_iter == output_channels:
            pe_output_channel_iter = 0
            base_output_channel_iter += 1
        bias_string += ","
    bias_string = bias_string[:-1]
    #bias_string += "};"

    return bias_string

def fc_weight_mapper_2D(weights, PE, SIMD, weight_bits, weight_int_bits, float=False):
    """
    | Transforms weights of conv layer to a string that can be included in .hpp file.
    | Weight have shape: OUTPUT_CHANNELS x INPUT_CHANNELS x KERNEL_SIZE x KERNEL_SIZE.
    | Stores weights in ap_uint<input_channels*bit_width> OUTPUT_CHANNELS x KERNEL_SIZE*KERNEL_SIZE two dimensional array.
    | Binary representation of weight corresponds to ap_fixed number in HLS.
    | Therefore *reinterpret_cast* can be used in .cpp to access weight in ap_fixed format.

    :param weights: Weights of convolutional layer as given by *layer.weight*
    :param PE: Number of parallel computed output channels
    :param SIMD: Number of parallel computed input channels
    :param str i: index of layer
    :param int weight_bits: Number of bits to use for quantization
    :param int weight_int_bits: Number of integer bits to use for quantization
    :return: String to save in .hpp file
    """

    output_channels = weights.shape[0]
    input_channels = weights.shape[1]

    TILES = int( (input_channels / SIMD) * (output_channels / PE) )

    weight_string = ""

    base_output_channel_iter = 0
    pe_output_channel_iter = 0
    input_channel = 0

    for PE_iter in range(PE):
        if PE_iter != 0:
            weight_string += "\n"
        weight_string += "{"
        for tile in range(TILES):
            weight_string += '"0x' if float is not True else ''
            bin_string = ""
            for SIMD_iter in reversed(range(SIMD)):
                bin_string += float_to_binary_string(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel], weight_bits, weight_int_bits) if float is not True else str(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel].item())
                if SIMD_iter == 0:
                    input_channel += SIMD
                    if input_channel == input_channels:
                        input_channel = 0
                        pe_output_channel_iter += PE
                        if pe_output_channel_iter == output_channels:
                            pe_output_channel_iter = 0
                            base_output_channel_iter += 1

            weight_string += format((int(bin_string, 2)), 'x') if float is not True else bin_string
            weight_string += '",' if float is not True else ','
        weight_string = weight_string[:-1]
        weight_string += "}"
        weight_string += ","
    weight_string = weight_string[:-1]

    return weight_string

def conv_weight_mapper_2D(weights, PE, SIMD, weight_bits, weight_int_bits, float=False):
    """
    | Transforms weights of conv layer to a string that can be included in .hpp file.
    | Weight have shape: OUTPUT_CHANNELS x INPUT_CHANNELS x KERNEL_SIZE x KERNEL_SIZE.
    | Stores weights in ap_uint<input_channels*bit_width> OUTPUT_CHANNELS x KERNEL_SIZE*KERNEL_SIZE two dimensional array.
    | Binary representation of weight corresponds to ap_fixed number in HLS.
    | Therefore *reinterpret_cast* can be used in .cpp to access weight in ap_fixed format.

    :param weights: Weights of convolutional layer as given by *layer.weight*
    :param PE: Number of parallel computed output channels
    :param SIMD: Number of parallel computed input channels
    :param str i: index of layer
    :param int weight_bits: Number of bits to use for quantization
    :param int weight_int_bits: Number of integer bits to use for quantization
    :param bool float: Flag indicating if the model should be treated as a float model, if set the weights are printed as it is with out converting to hexdecimal
    :return: String to save in .hpp file
    """

    output_channels = weights.shape[0]
    input_channels = weights.shape[1]
    kernel_size = weights.shape[2]

    TILES = int(kernel_size * kernel_size * (input_channels / SIMD) * (output_channels / PE))

    weight_string = ""

    base_output_channel_iter = 0
    pe_output_channel_iter = 0
    input_channel = 0
    dim1 = 0
    dim2 = 0

    for PE_iter in range(PE):
        if PE_iter != 0:
            weight_string += "\n"
        weight_string += "{"
        for tile in range(TILES):
            weight_string += '"0x' if float is not True else ''
            bin_string = ""
            for SIMD_iter in range(SIMD):
                if not float:
                    bin_string += float_to_binary_string(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel][dim2][dim1], weight_bits, weight_int_bits)
                else:
                    bin_string += str(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel][dim2][dim1].item())
                    if SIMD > 1 and SIMD_iter != SIMD - 1:
                        bin_string += ", "
                if SIMD_iter == SIMD - 1:
                    input_channel += SIMD
                    if input_channel == input_channels:
                        input_channel = 0
                        dim1 += 1
                        if dim1 == kernel_size:
                            dim1 = 0
                            dim2 += 1
                            if dim2 == kernel_size:
                                dim2 = 0
                                pe_output_channel_iter += PE
                                if pe_output_channel_iter == output_channels:
                                    pe_output_channel_iter = 0
                                    base_output_channel_iter += 1

            weight_string += format((int(bin_string, 2)), 'x') if float is not True else bin_string
            weight_string += '",' if float is not True else ','
        weight_string = weight_string[:-1]
        weight_string += "}"
        weight_string += ","
    weight_string = weight_string[:-1]

    return weight_string

def tr_conv_weight_mapper_2D(weights, PE, SIMD, weight_bits, weight_int_bits, float=False):
    """
    | Transforms weights of transposed conv layer to a string that can be included in .hpp file.
    | Weight have shape: INPUT_CHANNELS x OUTPUT_CHANNELS x KERNEL_SIZE x KERNEL_SIZE.
    | Stores weights in ap_uint<input_channels*bit_width> OUTPUT_CHANNELS x KERNEL_SIZE*KERNEL_SIZE two dimensional array.
    | Binary representation of weight corresponds to ap_fixed number in HLS.
    | Therefore *reinterpret_cast* can be used in .cpp to access weight in ap_fixed format.

    :param weights: Weights of transposed convolutional layer as given by *layer.weight*
    :param PE: Number of parallel computed output channels
    :param SIMD: Number of parallel computed input channels
    :param str i: index of layer
    :param int weight_bits: Number of bits to use for quantization
    :param int weight_int_bits: Number of integer bits to use for quantization
    :return: String to save in .hpp file
    """

    input_channels = weights.shape[0]
    output_channels = weights.shape[1]
    kernel_size = weights.shape[2]

    TILES = int(kernel_size * kernel_size * (input_channels / SIMD) * (output_channels / PE))

    weight_string = ""

    base_output_channel_iter = 0
    pe_output_channel_iter = 0
    input_channel = 0
    dim1 = 0
    dim2 = 0

    for PE_iter in range(PE):
        if PE_iter != 0:
            weight_string += "\n"
        weight_string += "{"
        for tile in range(TILES):
            weight_string += '"0x' if float is not True else ''
            bin_string = ""
            for SIMD_iter in range(SIMD):
                in_ch_idx = SIMD_iter + input_channel
                out_ch_idx = base_output_channel_iter + pe_output_channel_iter
                if float:
                    bin_string += str(int(weights[in_ch_idx][out_ch_idx][dim2][dim1].item()))
                else:
                    bin_string += float_to_binary_string(weights[in_ch_idx][out_ch_idx][dim2][dim1], weight_bits, weight_int_bits)
                if SIMD > 1 and SIMD_iter != SIMD - 1:
                    bin_string += ", "
            input_channel += SIMD
            if input_channel == input_channels:
                input_channel = 0
                pe_output_channel_iter += PE
                if pe_output_channel_iter >= output_channels:
                    pe_output_channel_iter = 0
                    dim1 += 1
                    if dim1 == kernel_size:
                        dim1 = 0
                        dim2 += 1
                        if dim2 == kernel_size:
                            dim2 = 0
                            base_output_channel_iter += 1

            weight_string += format((int(bin_string, 2)), 'x') if float is not True else bin_string
            weight_string += '",' if float is not True else ','
        weight_string = weight_string[:-1]
        weight_string += "}"
        weight_string += ","
    weight_string = weight_string[:-1]

    return weight_string
##############################################################################################################
# Profiler
##############################################################################################################

class Profiler(object):
    def __init__(self, name):
        super(Profiler, self).__init__()
        self.minimum = 1000.0
        self.maximum = -1000.0
        self.int_bit_width = 0
        self.name = name

    def update(self, input_):
        minimum = torch.min(input_)
        maximum = torch.max(input_)

        if (minimum < self.minimum):
            self.minimum = minimum

        if (maximum > self.maximum):
            self.maximum = maximum

    def status(self):
        return minimum, maximum

    def reset(self):
        self.minimum = 1000.0
        self.maximum = -1000.0
        self.int_bit_width = 0

    def return_int_bit_width(self):
        minimum = int(self.minimum)
        maximum = int(self.maximum)
        minimum = int(math.log2(math.fabs(minimum))) + 1 if minimum != 0.0 else 0
        maximum = int(math.log2(math.fabs(maximum))) + 1 if maximum != 0.0 else 0
        self.int_bit_width = minimum if minimum > maximum else maximum
        return self.int_bit_width

    def __repr__(self):

        return self.__class__.__name__ + ': ' \
        + self.name \
        + ': Min=' + str(self.minimum.item()) \
        + ', Max=' + str(self.maximum.item()) \
        + ', int_bit_width=' + str(self.return_int_bit_width())

##############################################################################################################
# Quantization primitives
##############################################################################################################

def quantize(x, params):
    q_tensor = x * params.prescale

    q_tensor_r = q_tensor.round()
    q_tensor_f = q_tensor.floor()
    q_tensor_c = q_tensor.ceil()
    q_tensor_tmp = torch.where(q_tensor-q_tensor.int()==0.5,q_tensor_f,q_tensor_r)
    q_tensor_tmp = torch.where(q_tensor-q_tensor.int()==-0.5,q_tensor_c,q_tensor_tmp)

    q_tensor = q_tensor_tmp * params.postscale
    q_tensor = q_tensor.clamp(params.min_val, params.max_val)
    return q_tensor

class Identity(Function):

    '''@staticmethod
    def symbolic(g, input):
        return g.op('Identity', input)'''

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad

class QuantizeFixedActivation(Function):

    '''@staticmethod
    def symbolic(g, params, input):
        return g.op('QuantizeFixedActivation', input)'''

    @staticmethod
    def forward(ctx, params, a_in):
        ctx.save_for_backward(a_in)
        ctx.params = params
        a_out = quantize(a_in, params)
        return a_out

    @staticmethod
    def backward(ctx, grad_output):
        params = ctx.params
        a_in, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(a_in.ge(params.max_val) | a_in.le(params.min_val), 0)
        return None, grad_input

'''class QuantizeFixedActivation(nn.Module):

    def forward(self, params, input):
        return partial(QuantizeFixedActivationFunction.apply, params)'''

class QuantizeFixedWeight(Function):

    '''@staticmethod
    def symbolic(g, params, input):
        return g.op('QuantizeFixedWeight', input)'''

    @staticmethod
    def forward(ctx, params, w_in):
        w_out = quantize(w_in, params)
        return w_out

    @staticmethod
    def backward(ctx, grad):
        return None, grad

class BinarizeWeight(Function):
    '''
    Binarize weights with output channel-wise scale for Linear
    Binarize weights with layer-wise scale for Conv2d
    '''
    @staticmethod
    def forward(ctx, w_in):
        if w_in.dim() == 2:
            scale = w_in.abs().mean(dim=1)
        else:
            scale = w_in.abs().mean()
        w_out = scale * w_in.sign()
        return w_out

    @staticmethod
    def backward(ctx, grad):
        return grad

class BinarizeActivation(Function):

    @staticmethod
    def forward(ctx, a_in):
        ctx.save_for_backward(a_in)
        a_out = a_in.sign()
        return a_out

    @staticmethod
    def backward(ctx, grad_output):
        a_in, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(a_in.ge(1) | a_in.le(-1), 0)
        return grad_input

class QuantizationParams():

    def __init__(self, q_type):
        self.q_type = q_type

class SignedFixedQuantizationParams(QuantizationParams):

    def __init__(self, bit_width, int_bit_width, q_type):
        super(SignedFixedQuantizationParams, self).__init__(q_type)
        self.bit_width = bit_width
        self.int_bit_width = int_bit_width  # including implicit sign bit

        self.frac_bit_width = self.bit_width - self.int_bit_width
        self.prescale = 2 ** self.frac_bit_width
        self.postscale = 2 ** (- self.frac_bit_width)
        self.min_val = - (2 ** (self.int_bit_width - 1))
        self.max_val = - self.min_val - self.postscale

class UnsignedFixedQuantizationParams(QuantizationParams):

    def __init__(self, bit_width, int_bit_width, q_type):
        super(UnsignedFixedQuantizationParams, self).__init__(q_type)
        self.bit_width = bit_width
        self.int_bit_width = int_bit_width

        self.frac_bit_width = self.bit_width - self.int_bit_width
        self.prescale = 2 ** self.frac_bit_width
        self.postscale = 2 ** (- self.frac_bit_width)
        self.min_val = 0.0
        self.max_val = 2 ** self.int_bit_width - self.postscale

class QuantizationScheme(nn.Module):

    def __init__(self, q_type, threshold=None, bit_width=None, int_bit_width=None):
        super(QuantizationScheme, self).__init__()
        if q_type == 'identity':
            self.q_params = QuantizationParams(q_type)
        elif q_type == 'fixed_unsigned':
            self.q_params = UnsignedFixedQuantizationParams(bit_width, int_bit_width, q_type)
        elif q_type == 'fixed_signed':
            self.q_params = SignedFixedQuantizationParams(bit_width, int_bit_width, q_type)
        elif q_type == 'binary':
            self.q_params = QuantizationParams(q_type)
        else:
            raise Exception('Unknown quantization scheme: {}.'.format(q_type))

    def __repr__(self):
        if self.q_params.q_type == 'binary' or self.q_params.q_type == 'identity':
            string = "q_type: {}".format(self.q_params.q_type)
        else:
            string = "q_type: {}, bit_width: {}, int_bit_width: {}".format(self.q_params.q_type, self.q_params.bit_width, self.q_params.int_bit_width)
        return string

class WeightQuantizationScheme(QuantizationScheme):

    def __init__(self, q_type, threshold=None, bit_width=None, int_bit_width=None):
        super(WeightQuantizationScheme, self).__init__(q_type, threshold, bit_width, int_bit_width)
        if self.q_params.q_type == 'identity':
            self.q_function = partial(Identity.apply)
        elif self.q_params.q_type == 'fixed_unsigned' or self.q_params.q_type == 'fixed_signed':
            self.q_function = partial(QuantizeFixedWeight.apply, self.q_params)
        elif self.q_params.q_type == 'binary':
            self.q_function = partial(BinarizeWeight.apply)
        else:
            raise Exception('Unknown quantization scheme: {}.'.format(q_type))

    def forward(self, x):
        return self.q_function(x)

class ActivationQuantizationScheme(QuantizationScheme):

    def __init__(self, q_type, threshold=None, bit_width=None, int_bit_width=None):
        super(ActivationQuantizationScheme, self).__init__(q_type, threshold, bit_width, int_bit_width)
        if self.q_params.q_type == 'identity':
            self.q_function = partial(Identity.apply)
        elif self.q_params.q_type == 'fixed_unsigned' or self.q_params.q_type == 'fixed_signed':
            self.q_function = partial(QuantizeFixedActivation.apply, self.q_params)
        elif self.q_params.q_type == 'binary':
            self.q_function = partial(BinarizeActivation.apply)
        else:
            raise Exception('Unknown quantization scheme: {}.'.format(q_type))

    def forward(self, x):
        return self.q_function(x)

class BiasQuantizationScheme(QuantizationScheme):

    def __init__(self, q_type, threshold=None, bit_width=None, int_bit_width=None):
        super(BiasQuantizationScheme, self).__init__(q_type, threshold, bit_width, int_bit_width)
        if self.q_params.q_type == 'identity':
            self.q_function = partial(Identity.apply)
        elif self.q_params.q_type == 'fixed_unsigned' or self.q_params.q_type == 'fixed_signed':
            self.q_function = partial(QuantizeFixedWeight.apply, self.q_params)
        else:
            raise Exception('Unknown quantization scheme: {}.'.format(q_type))

    def forward(self, x):
        return self.q_function(x)

##############################################################################################################
##############################################################################################################

class QuantizationConfig():
    def __init__(self, weight_quantization, bias_quantization, in_activation_quantization, out_activation_quantization):
        self.weight_quantization = weight_quantization
        self.bias_quantization = bias_quantization
        self.in_activation_quantization = in_activation_quantization
        self.out_activation_quantization = out_activation_quantization

    def __repr__(self):
        q_type = self.weight_quantization.q_params.q_type
        if q_type == 'binary' or q_type == 'identity':
            string = "weight_quantization: q_type: {}".format(q_type)
        else:
            int_bit_width = self.weight_quantization.q_params.int_bit_width
            bit_width = self.weight_quantization.q_params.bit_width
            string = "weight_quantization: q_type: {}, bit_width: {}, int_bit_width: {}".format(q_type, bit_width, int_bit_width)

        q_type = self.bias_quantization.q_params.q_type
        if q_type == 'binary' or q_type == 'identity':
            string = string + "\n" + "bias_quantization: q_type: {}".format(q_type)
        else:
            int_bit_width = self.bias_quantization.q_params.int_bit_width
            bit_width = self.bias_quantization.q_params.bit_width
            string = string + "\n" + "bias_quantization: q_type: {}, bit_width: {}, int_bit_width: {}".format(q_type, bit_width, int_bit_width)

        q_type = self.in_activation_quantization.q_params.q_type
        if q_type == 'binary' or q_type == 'identity':
            string = string + "\n" + "in_activation_quantization: q_type: {}".format(q_type)
        else:
            int_bit_width = self.in_activation_quantization.q_params.int_bit_width
            bit_width = self.in_activation_quantization.q_params.bit_width
            string = string + "\n" + "in_activation_quantization: q_type: {}, bit_width: {}, int_bit_width: {}".format(q_type, bit_width, int_bit_width)

        q_type = self.out_activation_quantization.q_params.q_type
        if q_type == 'binary' or q_type == 'identity':
            string = string + "\n" + "out_activation_quantization: q_type: {}".format(q_type)
        else:
            int_bit_width = self.out_activation_quantization.q_params.int_bit_width
            bit_width = self.out_activation_quantization.q_params.bit_width
            string = string + "\n" + "out_activation_quantization: q_type: {}, bit_width: {}, int_bit_width: {}".format(q_type, bit_width, int_bit_width)

        return string

##############################################################################################################
# Qunatized layers
##############################################################################################################

########################################################################
# Qunatized ReLU6 layer

class QReLU6(nn.ReLU6):
    def __init__(self, inplace=False):
        super(QReLU6, self).__init__()

        self.max_val = 6
        int_bit_width = 4
        bit_width = 8
        self.quantize_out_activations = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        self.profiler = [Profiler("Input"), Profiler("Output")]
        quantizer = ActivationQuantizationScheme(q_type='identity')
        self.quantizer = []
        for _ in self.profiler:
            self.quantizer.append(quantizer)

    def forward(self, input):

        if self.training is False:
            input = self.quantizer[0](input)
            self.profiler[0].update(input)

        output = F.hardtanh(input, self.min_val, self.max_val, self.inplace)

        output = self.quantize_out_activations(output)

        if self.training is False:
            output = self.quantizer[1](output)
            self.profiler[1].update(output)

        return output

    def print_profiler(self):
        for p in self.profiler:
            print(p)

    def reset_profiler(self):
        for p in self.profiler:
            p.reset()

    def update_precisions(self, previous_layer):
        int_bit_width = self.profiler[0].return_int_bit_width() + 1
        bit_width = int_bit_width + 8
        self.quantizer[0] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        int_bit_width = 4
        bit_width = 8
        self.quantizer[1] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        self.reset_profiler()

########################################################################
# Qunatized Split layer

class QSplit(nn.Module):
    def __init__(self):
        super(QSplit, self).__init__()

        self.profiler = [Profiler("Input")]
        quantizer = ActivationQuantizationScheme(q_type='identity')
        self.quantizer = []
        for _ in self.profiler:
            self.quantizer.append(quantizer)

    def forward(self, input):

        if self.training is False:
            input = self.quantizer[0](input)
            self.profiler[0].update(input)

        output = input

        return output

    def print_profiler(self):
        for p in self.profiler:
            print(p)

    def reset_profiler(self):
        for p in self.profiler:
            p.reset()

    def update_precisions(self, previous_layer):
        for i, p in enumerate(self.profiler):
            int_bit_width = p.return_int_bit_width() + 1
            bit_width = int_bit_width + 8
            self.quantizer[i] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)
        self.reset_profiler()

########################################################################
# Qunatized Add/Merge layer

class QAdd(nn.Module):
    def __init__(self):
        super(QAdd, self).__init__()

        self.profiler = [Profiler("ShortCut"), Profiler("Input"), Profiler("Output")]
        quantizer = ActivationQuantizationScheme(q_type='identity')
        self.quantizer = []
        for _ in self.profiler:
            self.quantizer.append(quantizer)

    def forward(self, short_cut, input):

        if self.training is False:
            short_cut = self.quantizer[0](short_cut)
            self.profiler[0].update(short_cut)

            input = self.quantizer[1](input)
            self.profiler[1].update(input)

        output = short_cut + input

        if self.training is False:
            output = self.quantizer[2](output)
            self.profiler[2].update(output)

        return output

    def print_profiler(self):
        for p in self.profiler:
            print(p)

    def reset_profiler(self):
        for p in self.profiler:
            p.reset()

    def update_precisions(self, previous_layer):
        for i, p in enumerate(self.profiler):
            int_bit_width = p.return_int_bit_width() + 1
            bit_width = int_bit_width + 8
            self.quantizer[i] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)
        self.reset_profiler()

########################################################################
# Qunatized BatchNorm2d layer

class QBatchNorm2d(nn.BatchNorm2d):
    """
    Quantized 2D BatchNorm Layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        nn.BatchNorm2d.__init__(self, num_features, eps, momentum, affine, track_running_stats)

        self.profiler = [Profiler("Input"), Profiler("Scale"), Profiler("Shift"), Profiler("Accumulator"), Profiler("Output")]
        quantizer = ActivationQuantizationScheme(q_type='identity')
        self.quantizer = []
        for _ in self.profiler:
            self.quantizer.append(quantizer)

    def forward(self, input):

        if self.training is False:
            shape = (1, input.size()[1], 1 ,1)

            #input = self.quantizer[0](input)
            self.profiler[0].update(input)

            scale = self.weight.view(shape) / torch.sqrt(self.running_var.view(shape) + self.eps)
            scale = self.quantizer[1](scale)
            self.profiler[1].update(scale)

            shift = self.running_mean.view(shape) * scale - self.bias.view(shape)
            shift = self.quantizer[2](shift)
            self.profiler[2].update(shift)

            #output = input * scale - shift
            self.profiler[3].update(input)
            self.profiler[3].update(scale)
            self.profiler[3].update(shift)
            scaled = self.quantizer[3](input) * self.quantizer[3](scale)
            scaled = self.quantizer[3](scaled)
            self.profiler[3].update(scaled)
            shifted = scaled - self.quantizer[3](shift)
            output = self.quantizer[3](shifted)
            self.profiler[3].update(output)

            # do not quantize - the value will be quantized by next layer
            #output = self.quantizer[4](shifted)
            self.profiler[4].update(output)

        else:
            output = F.batch_norm(input, self.running_mean, self.running_var,
                                  self.weight, self.bias,
                                  self.training or not self.track_running_stats,
                                  self.momentum, self.eps)

        return output

    def print_profiler(self):
        for p in self.profiler:
            print(p)

    def reset_profiler(self):
        for p in self.profiler:
            p.reset()

    def update_precisions(self, previous_layer):
        for i, p in enumerate(self.profiler):
            int_bit_width = p.return_int_bit_width() + 1
            bit_width = int_bit_width + 8
            self.quantizer[i] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)
        self.reset_profiler()

########################################################################
# Qunatized BatchNorm1d layer

class QBatchNorm1d(nn.BatchNorm1d):
    """
    Quantized 1D BatchNorm Layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        nn.BatchNorm1d.__init__(self, num_features, eps, momentum, affine, track_running_stats)

        self.profiler = [Profiler("Input"), Profiler("Scale"), Profiler("Shift"), Profiler("Accumulator"), Profiler("Output")]
        quantizer = ActivationQuantizationScheme(q_type='identity')
        self.quantizer = []
        for _ in self.profiler:
            self.quantizer.append(quantizer)


    def forward(self, input):
        if self.training is False:
            shape = (1, input.size()[1])

            #input = self.quantizer[0](input)
            self.profiler[0].update(input)

            scale = self.weight.view(shape) / torch.sqrt(self.running_var.view(shape) + self.eps)
            scale = self.quantizer[1](scale)
            self.profiler[1].update(scale)

            shift = self.running_mean.view(shape) * scale - self.bias.view(shape)
            shift = self.quantizer[2](shift)
            self.profiler[2].update(shift)

            #output = input * scale - shift
            self.profiler[3].update(input)
            self.profiler[3].update(scale)
            self.profiler[3].update(shift)
            scaled = self.quantizer[3](input) * self.quantizer[3](scale)
            scaled = self.quantizer[3](scaled)
            self.profiler[3].update(scaled)
            shifted = scaled - self.quantizer[3](shift)
            output = self.quantizer[3](shifted)
            self.profiler[3].update(output)

            # do not quantize - the value will be quantized by next layer
            #output = self.quantizer[4](shifted)
            self.profiler[4].update(output)
        else:
            output = F.batch_norm(input, self.running_mean, self.running_var,
                                  self.weight, self.bias,
                                  self.training or not self.track_running_stats,
                                  self.momentum, self.eps)

        return output

    def print_profiler(self):
        for p in self.profiler:
            print(p)

    def reset_profiler(self):
        for p in self.profiler:
            p.reset()

    def update_precisions(self, previous_layer):
        for i, p in enumerate(self.profiler):
            int_bit_width = p.return_int_bit_width() + 1
            bit_width = int_bit_width + 8
            self.quantizer[i] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)
        self.reset_profiler()

########################################################################
# Qunatized Linear layer

class QLinear(nn.Linear):
    """
    Quantized Fully-Connected Layer
    """
    def __init__(self, q_config, *args, **kwargs):
        super(QLinear, self).__init__(*args, **kwargs)

        self.quantize_weight = q_config.weight_quantization
        self.quantize_bias = q_config.bias_quantization
        self.quantize_in_activation = q_config.in_activation_quantization
        self.quantize_out_activation = q_config.out_activation_quantization

        self.profiler = [Profiler("Input"), Profiler("Accumulator"), Profiler("Output")]
        quantizer = ActivationQuantizationScheme(q_type='identity')
        self.quantizer = []
        for _ in self.profiler:
            self.quantizer.append(quantizer)

    def forward(self, input):

        input = self.quantize_in_activation(input)

        if self.training is False:
            input = self.quantizer[0](input)
            self.profiler[0].update(input)

        quantized_weight = self.quantize_weight(self.weight)

        if self.bias is not None:
            quantized_bias = self.quantize_bias(self.bias)
        else:
            quantized_bias = self.bias

        output = F.linear(input, quantized_weight, quantized_bias)

        output = self.quantize_out_activation(output)

        if self.training is False:
            output = self.quantizer[1](output)
            self.profiler[1].update(output)

            # do not quantize - the value will be quantized by next layer
            #output = self.quantizer[2](output)
            self.profiler[2].update(output)

        return output

    def print_profiler(self):
        for p in self.profiler:
            print(p)

    def reset_profiler(self):
        for p in self.profiler:
            p.reset()

    def update_precisions(self, previous_layer):

        int_bit_width = self.profiler[0].return_int_bit_width() + 1
        bit_width = int_bit_width + 8
        self.quantizer[0] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        # does not prevent overflow
        #int_bit_width = self.profiler[1].return_int_bit_width() + 1
        #bit_width = int_bit_width + 8
        #self.quantizer[1] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        # constrain accumulator to 16.16 to avoid overflow
        int_bit_width = 16
        bit_width = 32
        self.quantizer[1] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        int_bit_width = self.profiler[2].return_int_bit_width() + 1
        bit_width = int_bit_width + 8
        self.quantizer[2] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        self.reset_profiler()

########################################################################
# Qunatized Conv2d layer

class QConv2d(nn.Conv2d):
    """
    Quantized 2D Convolutional Layer
    """
    def __init__(self, q_config, *args, **kwargs):
        super(QConv2d, self).__init__(*args, **kwargs)

        self.quantize_weight = q_config.weight_quantization
        self.quantize_bias = q_config.bias_quantization
        self.quantize_in_activation = q_config.in_activation_quantization
        self.quantize_out_activation = q_config.out_activation_quantization

        self.profiler = [Profiler("Input"), Profiler("Accumulator"), Profiler("Output")]
        quantizer = ActivationQuantizationScheme(q_type='identity')
        self.quantizer = []
        for _ in self.profiler:
            self.quantizer.append(quantizer)

        self.in_width = None
        self.in_height = None
        self.in_dim = None
        self.out_width = None
        self.out_height = None
        self.out_dim = None


    def forward(self, input):

        input = self.quantize_in_activation(input)

        if self.training is False:
            self.in_height = input.size()[2]
            self.in_width = input.size()[3]
            if self.in_height == self.in_width:
                self.in_dim = self.in_width

            input = self.quantizer[0](input)
            self.profiler[0].update(input)

        quantized_weight = self.quantize_weight(self.weight)

        if self.bias is not None:
            quantized_bias = self.quantize_bias(self.bias)
        else:
            quantized_bias = self.bias

        output = F.conv2d(input, quantized_weight, quantized_bias, self.stride, self.padding, self.dilation, self.groups)

        output = self.quantize_out_activation(output)

        if self.training is False:
            self.out_height = output.size()[2]
            self.out_width = output.size()[3]
            if self.out_height == self.out_width:
                self.out_dim = self.out_width

            output = self.quantizer[1](output)
            self.profiler[1].update(output)

            # do not quantize - the value will be quantized by next layer
            #output = self.quantizer[2](output)
            self.profiler[2].update(output)

        return output

    def print_profiler(self):
        for p in self.profiler:
            print(p)

    def reset_profiler(self):
        for p in self.profiler:
            p.reset()

    def update_precisions(self, previous_layer):
        # for the very first conv layer
        if self.quantize_in_activation.q_params.q_type == 'fixed_signed':
            int_bit_width = self.quantize_in_activation.q_params.int_bit_width
            bit_width = self.quantize_in_activation.q_params.bit_width
            self.quantizer[0] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)
        elif isinstance(previous_layer, nn.ReLU6):
            int_bit_width = 4
            bit_width = 8
            self.quantizer[0] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)
        else:
            int_bit_width = self.profiler[0].return_int_bit_width() + 1
            bit_width = int_bit_width + 8
            self.quantizer[0] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        # does not prevent overflow
        #int_bit_width = self.profiler[1].return_int_bit_width() + 1
        #bit_width = int_bit_width + 8
        #self.quantizer[1] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        # constrain accumulator to 16.16 to avoid overflow
        int_bit_width = 16
        bit_width = 32
        self.quantizer[1] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        int_bit_width = self.profiler[2].return_int_bit_width() + 1
        bit_width = int_bit_width + 8
        self.quantizer[2] = ActivationQuantizationScheme(q_type='fixed_signed', bit_width=bit_width, int_bit_width=int_bit_width)

        self.reset_profiler()












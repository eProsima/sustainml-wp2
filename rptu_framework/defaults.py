from enum import Enum

class SupportedNodes:
    conv = 'Conv_layer'
    tr_conv = 'ConvTranspose'
    linear = 'Linear'
    bn = 'Bn'  # BatchNormalization
    ins = 'In'  # InstanceNormalization
    relu = 'Relu'
    relu6 = 'Relu6'
    merge = 'Merge'
    split = 'Split'
    upsample = 'UpSample'
    reshape = 'Reshape'
    transpose = 'Transpose'
    quantize = 'QuantizeLinear'
    dequantize = 'DequantizeLinear'
    clip = 'Clip'
    maxpool = 'MaxPool'
    constant = 'Constant'
    mul = 'Mul'
    sigmoid = 'Sigmoid'
    hardswish = 'HardSwish'
    globalavgpool = 'GlobalAveragePool'
    expand = 'Expand'
    concat = 'Concat'
    sdwc = 'sdwc'  # StreamingDataWidthConverter_Batch
    
    @staticmethod
    def is_supported(node_type):
        return node_type in SupportedNodes.__dict__.values()

class DefaultValues:
    bits = 32
    int_bits = 16
    simd = 1
    pe = 1
    freq_mhz = 225e6
    target_fps = 30
    min_act_value = -128  # Used for Onnx Int8 Quantization
    max_act_value = 127  # Used for Onnx Int8 Quantization
    brevitas_default_act_bits = 8
    brevitas_default_act_int_bits = 1
    int_quant_fixed_scale_bits = 32

class Dtypes(Enum):
    FLOAT = 'float'
    FIXED = 'fixed'
    INT = 'int'

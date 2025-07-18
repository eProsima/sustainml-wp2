"""
This module defines the NodeHLS class, which is responsible for representing each node in the internal graph built using networkx.
The class initializes node attributes, defines macros and data types, and
provides methods for generating instances and definations.
"""
import math
import torch

from .quantization import conv_weight_mapper_2D, conv_bias_mapper_2D, fc_weight_mapper_2D, WeightQuantizationScheme, BiasQuantizationScheme, tr_conv_weight_mapper_2D
from .utils import shift_bit_length

import numpy as np
from .defaults import SupportedNodes, Dtypes

class NodeHLS():
    def __init__(self, node, fixed_scale=False):
        self.name = node['name']
        self.op_type = node['op_type']
        self.input_shape = node['input_shape']
        self.output_shape = node['output_shape']
        self.pe = node['pe']
        self.simd = node['simd']
        self.dtype = node['dtype']
        self.fixed_scale = fixed_scale
        if SupportedNodes.conv in self.op_type or SupportedNodes.linear in self.op_type:
            self.max_pe = node['weight'].shape[0]
            self.max_simd = node['weight'].shape[1]
        elif SupportedNodes.tr_conv in self.op_type:
            self.max_pe = node['weight'].shape[1]
            self.max_simd = node['weight'].shape[0]
        else:
            self.max_pe = math.inf  # A workaround to make sure the comparision in networkhls.set_parallelism works
            self.max_simd = math.inf
        self._prop_input_stream_width = None
        self._prop_output_stream_width = None
        self._prop_input_accesses = None
        self._prop_output_accesses = None

        if not SupportedNodes.is_supported(self.op_type):
            print(f"NodeHLS: __init__: unsupported layer {self.op_type}")
            return None

        # Suffix
        self.__kernel_size = "_K"
        self.__in_channels = "_IFM_CH"
        self.__in_dim = "_IFM_DIM"
        self.__out_channels = "_OFM_CH"
        self.__out_dim = "_OFM_DIM"
        self.__group = "_GROUPS"
        self.__stride = "_STRIDE"
        self.__padding = "_PADDING"
        self.__in_features = "_IF"
        self.__out_features = "_OF"
        self.__pe = "_PE"
        self.__simd = "_SIMD"
        self.__in_bits = "_IFM_BITS"
        self.__in_int_bits = "_IFM_INT_BITS"
        self.__in_bits_aligned = "_IFM_BITS_ALIGNED"
        self.__input_stream_width = "_INPUT_STREAM_WIDTH"
        self.__input_mem_width = "_INPUT_MEM_WIDTH"
        self.__input_accesses = "_INPUT_ACCESSESS"
        self.__weight_bits = "_WEIGHT_BITS"
        self.__weight_int_bits = "_WEIGHT_INT_BITS"
        self.__weight_simd_width = "_WEIGHT_SIMD_WIDTH"
        self.__weight_simd_pe_width = "_WEIGHT_SIMD_PE_WIDTH"
        self.__bias_bits = "_BIAS_BITS"
        self.__bias_int_bits = "_BIAS_INT_BITS"
        self.__acc_bits = "_ACC_BITS"
        self.__acc_int_bits = "_ACC_INT_BITS"
        self.__activation_tiles = "_ATILES"
        self.__weight_tiles = "_WTILES"
        self.__scale_bits = "_SCALE_BITS"
        self.__scale_int_bits = "_SCALE_INT_BITS"
        self.__shift_bits = "_SHIFT_BITS"
        self.__shift_int_bits = "_SHIFT_INT_BITS"
        self.__out_bits = "_OFM_BITS"
        self.__out_int_bits = "_OFM_INT_BITS"
        self.__out_bits_aligned = "_OFM_BITS_ALIGNED"
        self.__m0_bits = "_M0_BITS"
        self.__m0_int_bits = "_M0_INT_BITS"
        self.__zero_point_bits = "_ZERO_POINT_BITS"
        self.__zero_point_int_bits = "_ZERO_POINT_INT_BITS"
        self.__output_stream_width = "_OUTPUT_STREAM_WIDTH"
        self.__output_mem_width = "_OUTPUT_MEM_WIDTH"
        self.__output_accesses = "_OUPUT_ACCESSESS"
        self.__input_dtype = "_input_dtype"
        self.__weight_dtype = "_weight_dtype"
        self.__bias_dtype = "_bias_dtype"
        self.__scale_dtype = "_scale_dtype"
        self.__shift_dtype = "_shift_dtype"
        self.__acc_dtype = "_acc_dtype"
        self.__output_dtype = "_output_dtype"
        self.__m0_dtype = "_m0_dtype"
        self.__zero_point_dtype = "_zero_point_dtype"
        self.__weights = "_weights"
        self.__activations = "_activations"
        self.__max = "_MAX"
        self.__min = "_MIN"
        self.__m0 = "_M0"
        self.__shift = "_SHIFT"
        self.__input_zero_point = "_INPUT_ZERO_POINT"
        self.__output_zero_point = "_OUTPUT_ZERO_POINT"
        self.__input1_m0 = "_INPUT1_M0"
        self.__input1_shift = "_INPUT1_SHIFT"
        self.__input1_zero_point = "_INPUT1_ZERO_POINT"
        self.__input2_m0 = "_INPUT2_M0"
        self.__input2_shift = "_INPUT2_SHIFT"
        self.__input2_zero_point = "_INPUT2_ZERO_POINT"

        self._pe = self.macros_def(self.__pe)
        self._simd = self.macros_def(self.__simd)

        if any(i in self.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]):
            if any(op in self.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
                self.kernel_shape = node['kernel_shape']
                self.kernel_size = self.kernel_shape[0]
                self.in_channels = self.input_shape[1]
                self.in_dim = self.input_shape[2]
                self.out_channels = self.output_shape[1]
                self.out_dim = self.output_shape[2]
                self.group = node['group']
                self.strides = node['strides']
                self.stride = self.strides[0]
                self.pads = node['pads']
                self.padding = self.pads[0]
                self._kernel_size = self.macros_def(self.__kernel_size)
                self._in_channels = self.macros_def(self.__in_channels)
                self._in_dim = self.macros_def(self.__in_dim)
                self._out_channels = self.macros_def(self.__out_channels)
                self._out_dim = self.macros_def(self.__out_dim)
                self._group = self.macros_def(self.__group)
                self._stride = self.macros_def(self.__stride)
                self._padding = self.macros_def(self.__padding)
            if SupportedNodes.linear in self.op_type:
                self.in_features = self.input_shape[1]
                self.out_features = self.output_shape[1]
                self._in_features = self.macros_def(self.__in_features)
                self._out_features = self.macros_def(self.__out_features)
            self.bias = node['bias']
            if self.bias is not None:
                self.bias_bits = node['bias_bits']
                self.bias_int_bits = node['bias_int_bits']
                self._bias_bits = self.macros_def(self.__bias_bits)
                self._bias_int_bits = self.macros_def(self.__bias_int_bits)
                self._bias_dtype = self.dtype_def(self.__bias_dtype)
            if self.dtype == Dtypes.INT:
                self.m0 = node['M0']
                if not self.fixed_scale:
                    self.shift = node['shift']
                self.input_zero_point = node['input_zero_point']
                self.output_zero_point = node['output_zero_point']
                self._m0 = self.macros_def(self.__m0)
                if not self.fixed_scale:
                    self._shift = self.macros_def(self.__shift)
                self._input_zero_point = self.macros_def(self.__input_zero_point)
                self._output_zero_point = self.macros_def(self.__output_zero_point)

            self._activations = self.dtype_def(self.__activations)
            self.weight = node['weight']
            self.weight_bits = node['weight_bits']
            self.weight_int_bits = node['weight_int_bits']
            self.acc_bits = node['acc_bits']
            self.acc_int_bits = node['acc_int_bits']
            self._weight_bits = self.macros_def(self.__weight_bits)
            self._weight_int_bits = self.macros_def(self.__weight_int_bits)
            self._weight_simd_width = self.macros_def(self.__weight_simd_width)
            self._weight_simd_pe_width = self.macros_def(self.__weight_simd_pe_width)
            self._acc_bits = self.macros_def(self.__acc_bits)
            self._acc_int_bits = self.macros_def(self.__acc_int_bits)
            self._weight_tiles = self.macros_def(self.__weight_tiles)
            self._activation_tiles = self.macros_def(self.__activation_tiles)
            self._weight_dtype = self.dtype_def(self.__weight_dtype)
            self._acc_dtype = self.dtype_def(self.__acc_dtype)
            self._weights = self.dtype_def(self.__weights)
            if self.fixed_scale:
                self.m0_bits = node['m0_bits']
                self.m0_int_bits = node['m0_int_bits']
                self._m0_bits = self.macros_def(self.__m0_bits)
                self._m0_int_bits = self.macros_def(self.__m0_int_bits)
                self._m0_dtype = self.dtype_def(self.__m0_dtype)

        if any(i in self.op_type for i in [SupportedNodes.bn]):
            _norm = "_norm"
            self.epsilon = node['epsilon']
            self.norm_weight = node['norm_weight']
            self.norm_bias = node['norm_bias']
            self.running_var = node['running_var']
            self.running_mean = node['running_mean']
            self.scale_bits = node['scale_bits']
            self.scale_int_bits = node['scale_int_bits']
            self.shift_bits = node['shift_bits']
            self.shift_int_bits = node['shift_int_bits']
            self.norm_acc_bits = node['norm_acc_bits']
            self.norm_acc_int_bits = node['norm_acc_int_bits']
            self._scale_bits = self.macros_def(self.__scale_bits)
            self._scale_int_bits = self.macros_def(self.__scale_int_bits)
            self._shift_bits = self.macros_def(self.__shift_bits)
            self._shift_int_bits = self.macros_def(self.__shift_int_bits)
            self._norm_acc_bits = self.macros_def(_norm,self.__acc_bits)
            self._norm_acc_int_bits = self.macros_def(_norm,self.__acc_int_bits)
            self._scale_dtype = self.dtype_def(self.__scale_dtype)
            self._shift_dtype = self.dtype_def(self.__shift_dtype)
            self._norm_acc_dtype = self.dtype_def(_norm,self.__acc_dtype)
            self._activations = self.dtype_def(self.__activations)

        if any(i in self.op_type for i in [SupportedNodes.ins]):
            _norm = "_norm"
            self.scale = node['scale']
            self.shift = node['shift']
            self.scale_bits = node['scale_bits']
            self.scale_int_bits = node['scale_int_bits']
            self.shift_bits = node['shift_bits']
            self.shift_int_bits = node['shift_int_bits']
            self.norm_acc_bits = node['norm_acc_bits']
            self.norm_acc_int_bits = node['norm_acc_int_bits']
            self._scale_bits = self.macros_def(self.__scale_bits)
            self._scale_int_bits = self.macros_def(self.__scale_int_bits)
            self._shift_bits = self.macros_def(self.__shift_bits)
            self._shift_int_bits = self.macros_def(self.__shift_int_bits)
            self._norm_acc_bits = self.macros_def(_norm,self.__acc_bits)
            self._norm_acc_int_bits = self.macros_def(_norm,self.__acc_int_bits)
            self._scale_dtype = self.dtype_def(self.__scale_dtype)
            self._shift_dtype = self.dtype_def(self.__shift_dtype)
            self._norm_acc_dtype = self.dtype_def(_norm,self.__acc_dtype)
            self._activations = self.dtype_def(self.__activations)

        if SupportedNodes.merge in self.op_type:
            _direct = "_direct"
            _fifo = "_fifo"
            self.in_bits = node['in_bits'][0]
            self.in_int_bits = node['in_int_bits'][0]
            self.in_bits_aligned = shift_bit_length(self.in_bits)
            self.input_stream_width = int(self.in_bits * self.simd)
            self.input_mem_width = int(self.in_bits_aligned * self.simd)
            self._in_bits = self.macros_def(_direct,self.__in_bits)
            self._in_int_bits = self.macros_def(_direct,self.__in_int_bits)
            self._in_bits_aligned = self.macros_def(_direct,self.__in_bits_aligned)
            self._input_stream_width = self.macros_def(_direct,self.__input_stream_width)
            self._input_mem_width = self.macros_def(_direct,self.__input_mem_width)
            self._input_dtype = self.dtype_def(_direct,self.__input_dtype)

            self.fifo_in_bits = node['in_bits'][1]
            self.fifo_in_int_bits = node['in_int_bits'][1]
            self.fifo_in_bits_aligned = shift_bit_length(self.fifo_in_bits)
            self.fifo_input_stream_width = int(self.fifo_in_bits * self.simd)
            self.fifo_input_mem_width = int(self.fifo_in_bits_aligned * self.simd)
            self._fifo_in_bits = self.macros_def(_fifo,self.__in_bits)
            self._fifo_in_int_bits = self.macros_def(_fifo,self.__in_int_bits)
            self._fifo_in_bits_aligned = self.macros_def(_fifo, self.__in_bits_aligned)
            self._fifo_input_stream_width = self.macros_def(_fifo,self.__input_stream_width)
            self._fifo_input_mem_width = self.macros_def(_fifo, self.__input_mem_width)
            self._fifo_input_dtype = self.dtype_def(_fifo,self.__input_dtype)
            if self.dtype is Dtypes.INT:
                self.input1_m0 = node['input1_M0']
                self.input1_zero_point = node['input1_zero_point']
                self.input2_m0 = node['input2_M0']
                self.input2_zero_point = node['input2_zero_point']
                self.output_zero_point = node['output_zero_point']
                self._input1_m0 = self.macros_def(self.__input1_m0)
                self._input1_zero_point = self.macros_def(self.__input1_zero_point)
                self._input2_m0 = self.macros_def(self.__input2_m0)
                self._input2_zero_point = self.macros_def(self.__input2_zero_point)
                self._output_zero_point = self.macros_def(self.__output_zero_point)
                if not self.fixed_scale:
                    self.input1_shift = node['input1_shift']
                    self.input2_shift = node['input2_shift']
                    self._input1_shift = self.macros_def(self.__input1_shift)
                    self._input2_shift = self.macros_def(self.__input2_shift)
                else:
                    self.m0_bits = node['m0_bits']
                    self.m0_int_bits = node['m0_int_bits']
                    self._m0_bits = self.macros_def(self.__m0_bits)
                    self._m0_int_bits = self.macros_def(self.__m0_int_bits)
                    self._m0_dtype = self.dtype_def(self.__m0_dtype)

        if any(i in self.op_type for i in [SupportedNodes.relu, SupportedNodes.relu6, SupportedNodes.hardswish]):
            self._activations = self.dtype_def(self.__activations)

        if SupportedNodes.upsample in self.op_type:
            self.in_channels = self.input_shape[1]
            self.in_dim = self.input_shape[2]
            self.out_channels = self.output_shape[1]
            self.out_dim = self.output_shape[2]
            self._in_channels = self.macros_def(self.__in_channels)
            self._in_dim = self.macros_def(self.__in_dim)
            self._out_channels = self.macros_def(self.__out_channels)
            self._out_dim = self.macros_def(self.__out_dim)

        if SupportedNodes.maxpool in self.op_type:
            self.kernel_shape = node['kernel_shape']
            self.kernel_size = self.kernel_shape[0]
            self.strides = node['strides']
            self.stride = self.strides[0]
            self.pads = node['pads']
            self.padding = self.pads[0]
            self.in_channels = self.input_shape[1]
            self.in_dim = self.input_shape[2]
            self.out_channels = self.output_shape[1]
            self.out_dim = self.output_shape[2]
            self._kernel_size = self.macros_def(self.__kernel_size)
            self._stride = self.macros_def(self.__stride)
            self._padding = self.macros_def(self.__padding)
            self._in_channels = self.macros_def(self.__in_channels)
            self._in_dim = self.macros_def(self.__in_dim)
            self._out_channels = self.macros_def(self.__out_channels)
            self._out_dim = self.macros_def(self.__out_dim)

        if SupportedNodes.mul in self.op_type:
            _act = "_mul"
            self.scale = node['scale']
            self.scale = np.array([self.scale])
            self.scale_bits = node['scale_bits']
            self.scale_int_bits = node['scale_int_bits']
            self.act_acc_bits = node['act_acc_bits']
            self.act_acc_int_bits = node['act_acc_int_bits']
            self.in_channels = self.input_shape[1]
            self.in_dim = self.input_shape[2]
            self.out_channels = self.output_shape[1]
            self.out_dim = self.output_shape[2]
            self._in_channels = self.macros_def(self.__in_channels)
            self._in_dim = self.macros_def(self.__in_dim)
            self._out_channels = self.macros_def(self.__out_channels)
            self._out_dim = self.macros_def(self.__out_dim)
            self._scale_bits = self.macros_def(self.__scale_bits)
            self._scale_int_bits = self.macros_def(self.__scale_int_bits)
            self._act_acc_bits = self.macros_def(_act,self.__acc_bits)
            self._act_acc_int_bits = self.macros_def(_act,self.__acc_int_bits)
            self._scale_dtype = self.dtype_def(self.__scale_dtype)
            self._act_acc_dtype = self.dtype_def(_act,self.__acc_dtype)
            self._activations = self.dtype_def(self.__activations)

        if SupportedNodes.sigmoid in self.op_type:
            self.in_channels = self.input_shape[1]
            self.in_dim = self.input_shape[2]
            self.out_channels = self.output_shape[1]
            self.out_dim = self.output_shape[2]
            self._in_channels = self.macros_def(self.__in_channels)
            self._in_dim = self.macros_def(self.__in_dim)
            self._out_channels = self.macros_def(self.__out_channels)
            self._out_dim = self.macros_def(self.__out_dim)

        if any(i in self.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear, SupportedNodes.bn, SupportedNodes.ins, SupportedNodes.relu, SupportedNodes.relu6, SupportedNodes.split, SupportedNodes.upsample, SupportedNodes.maxpool, SupportedNodes.mul, SupportedNodes.sigmoid, SupportedNodes.globalavgpool, SupportedNodes.expand, SupportedNodes.clip, SupportedNodes.concat, SupportedNodes.sdwc]):
            self.in_bits = node['in_bits']
            self.in_int_bits = node['in_int_bits']
            self.in_bits_aligned = shift_bit_length(self.in_bits)
            self.input_stream_width = int(self.in_bits * self.simd)
            self.input_mem_width = int(self.in_bits_aligned * self.simd)
            self._in_bits = self.macros_def(self.__in_bits)
            self._in_int_bits = self.macros_def(self.__in_int_bits)
            self._in_bits_aligned = self.macros_def(self.__in_bits_aligned)
            self._input_stream_width = self.macros_def(self.__input_stream_width)
            self._input_mem_width = self.macros_def(self.__input_mem_width)
            self._input_dtype = self.dtype_def(self.__input_dtype)

        if SupportedNodes.globalavgpool in self.op_type:
            self.in_channels = self.input_shape[1]
            self.in_dim = self.input_shape[2]
            self.out_dim = self.output_shape[2]
            self.acc_bits = node['acc_bits']
            self.acc_int_bits = node['acc_int_bits']
            self._in_channels = self.macros_def(self.__in_channels)
            self._in_dim = self.macros_def(self.__in_dim)
            self._out_dim = self.macros_def(self.__out_dim)
            self._acc_bits = self.macros_def(self.__acc_bits)
            self._acc_int_bits = self.macros_def(self.__acc_int_bits)
            self._acc_dtype = self.dtype_def(self.__acc_dtype)
            if self.dtype is Dtypes.INT:
                self.m0 = node['M0']
                if not self.fixed_scale:
                    self.shift = node['shift']
                self.input_zero_point = node['input_zero_point']
                self.output_zero_point = node['output_zero_point']
                self._m0 = self.macros_def(self.__m0)
                if not self.fixed_scale:
                    self._shift = self.macros_def(self.__shift)
                self._input_zero_point = self.macros_def(self.__input_zero_point)
                self._output_zero_point = self.macros_def(self.__output_zero_point)
                if self.fixed_scale:
                    self.m0_bits = node['m0_bits']
                    self.m0_int_bits = node['m0_int_bits']
                    self._m0_bits = self.macros_def(self.__m0_bits)
                    self._m0_int_bits = self.macros_def(self.__m0_int_bits)
                    self._m0_dtype = self.dtype_def(self.__m0_dtype)

        if SupportedNodes.expand in self.op_type:
            self.in_channels = self.input_shape[1]
            self.in_dim = self.input_shape[2]
            self.out_dim = self.output_shape[2]
            self._in_channels = self.macros_def(self.__in_channels)
            self._in_dim = self.macros_def(self.__in_dim)
            self._out_dim = self.macros_def(self.__out_dim)

        if SupportedNodes.clip in self.op_type:
            self.in_channels = self.input_shape[1]
            self.in_dim = self.input_shape[2]
            self.max = node['max']
            self.min = node['min']
            self._in_channels = self.macros_def(self.__in_channels)
            self._in_dim = self.macros_def(self.__in_dim)
            self._max = self.macros_def(self.__max)
            self._min = self.macros_def(self.__min)

        if SupportedNodes.concat in self.op_type:
            self.in_channels = self.input_shape[1]
            self.in_dim = self.input_shape[2]
            self._in_channels = self.macros_def(self.__in_channels)
            self._in_dim = self.macros_def(self.__in_dim)
            if self.dtype is Dtypes.INT and self.fixed_scale:
                self._m_zp_dict = dict()
                for i, inp in enumerate(node['m_zp']):
                    self._m_zp_dict[self.macros_def(f'_{i}{self.__m0}')]= inp[0]
                    self._m_zp_dict[self.macros_def(f'_{i}{self.__input_zero_point}')]= inp[1]
                self.scale_bits = node['m_bits']
                self.scale_int_bits = node['m_int_bits']
                self.shift_bits = node['zero_point_bits']
                self.shift_int_bits = node['zero_point_int_bits']
                self._scale_bits = self.macros_def(self.__scale_bits)
                self._scale_int_bits = self.macros_def(self.__scale_int_bits)
                self._shift_bits = self.macros_def(self.__shift_bits)
                self._shift_int_bits = self.macros_def(self.__shift_int_bits)
                self._scale_dtype = self.dtype_def(self.__scale_dtype)
                self._shift_dtype = self.dtype_def(self.__shift_dtype)

        self.out_bits = node['out_bits']
        self.out_int_bits = node['out_int_bits']
        self.out_bits_aligned = shift_bit_length(self.out_bits)
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)
        self._out_bits = self.macros_def(self.__out_bits)
        self._out_int_bits = self.macros_def(self.__out_int_bits)
        self._out_bits_aligned = self.macros_def(self.__out_bits_aligned)
        self._output_stream_width = self.macros_def(self.__output_stream_width)
        self._output_mem_width = self.macros_def(self.__output_mem_width)
        self._output_dtype = self.dtype_def(self.__output_dtype)
        self.num_params = self.get_num_params()

        self._input_accesses = self.macros_def(self.__input_accesses)
        self._output_accesses = self.macros_def(self.__output_accesses)

        # This should be done last to override the default values
        if SupportedNodes.sdwc in self.op_type:
            # Since the sdwc is a simple node that depends on the macros of other functions there is no need to define seprate macros for it
            self._input_stream_width = node['_input_stream_width']
            self._output_stream_width = node['_output_stream_width']
            self._input_accesses = node['_input_accesses']

    @property
    def input_stream_width(self):
        return int(self.in_bits * self.simd)

    @input_stream_width.setter
    def input_stream_width(self, value):
        self._prop_input_stream_width = value

    @property
    def output_stream_width(self):
        return int(self.out_bits * self.pe)

    @output_stream_width.setter
    def output_stream_width(self, value):
        self._prop_output_stream_width = value

    @property
    def input_accesses(self):
        if self._prop_input_accesses is not None:
            return self._prop_input_accesses
        elif any(i in self.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.maxpool, SupportedNodes.upsample, SupportedNodes.expand, SupportedNodes.globalavgpool]):
            self._prop_input_accesses = int(self.in_bits * self.in_channels * self.in_dim * self.in_dim // self.input_stream_width)
        elif SupportedNodes.linear in self.op_type:
            self._prop_input_accesses = int(self.in_bits * self.in_features // self.input_stream_width)
        else:
            # If you're seeing this error, I guess the cat walked over my keyboard ¯\_(ツ)_/¯
            raise NotImplementedError(f"input_accesses: {self.op_type} - not implemented yet (or the cat deleted it)")
        return self._prop_input_accesses

    @input_accesses.setter
    def input_accesses(self, value):
        self._prop_input_accesses = value

    @property
    def output_accesses(self):
        if self._prop_output_accesses is not None:
            return self._prop_output_accesses
        elif any(i in self.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.maxpool, SupportedNodes.upsample]):
            self._prop_output_accesses = int(self.out_bits * self.out_channels * self.out_dim * self.out_dim // self.output_stream_width)
        elif SupportedNodes.linear in self.op_type:
            self._prop_output_accesses = int(self.out_bits * self.out_features // self.output_stream_width)
        elif SupportedNodes.expand in self.op_type:
            self._prop_output_accesses = int(self.out_bits * self.in_channels * self.out_dim * self.out_dim // self.output_stream_width)
        elif SupportedNodes.globalavgpool in self.op_type:
            self._prop_output_accesses = int(self.out_bits * self.in_channels // self.output_stream_width)
        else:
            # If you're seeing this error, I guess the cat walked over my keyboard ¯\_(ツ)_/¯
            raise NotImplementedError(f"output_accesses: {self.op_type} - not implemented yet (or the cat deleted it)")
        return self._prop_output_accesses

    @output_accesses.setter
    def output_accesses(self, value):
        self._prop_output_accesses = value

    def __str__(self):
        return f"NodeHLS: {self.name} {self.op_type}"

    def update_off_chip_outputs(self):
        self.out_bits_aligned = shift_bit_length(self.out_bits)
        self.out_bits = self.out_bits_aligned
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)

    def update_off_chip_inputs(self):
        self.in_bits_aligned = shift_bit_length(self.in_bits)
        self.in_bits = self.in_bits_aligned
        self.input_stream_width = int(self.in_bits * self.simd)
        self.input_mem_width = int(self.in_bits_aligned * self.simd)

    def get_input_accesses(self, _input_width):
        assert isinstance(_input_width, str), f'Input width should be str, got {type(_input_width)}'
        if any(op in self.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
            return f"{self._in_bits_aligned} * {self._in_channels} * {self._in_dim} * {self._in_dim}  / {_input_width}"
        elif SupportedNodes.linear in self.op_type:
            return f"{self._in_bits_aligned} * {self._in_features} / {_input_width}"

    def get_output_accesses(self, _output_width):
        assert isinstance(_output_width, str), f'Output width should be str, got {type(_output_width)}'
        if any(op in self.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
            return f"{self._out_bits_aligned} * {self._out_channels} * {self._out_dim} * {self._out_dim}  / {_output_width}"
        elif SupportedNodes.linear in self.op_type:
            return f"{self._out_bits_aligned} * {self._out_features} / {_output_width}"
        elif any(op in self.op_type for op in [SupportedNodes.merge, SupportedNodes.split]):
            return f"{self._out_bits_aligned} * {self._output_accesses} / {_output_width}"
        else:
            raise NotImplementedError(f"get_output_accesses: {self.op_type}")

    def get_instance_layer(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        if any(op in self.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
            instance += f"\t{self._kernel_size},\n"
            instance += f"\t{self._in_channels},\n"
            instance += f"\t{self._in_dim},\n"
            instance += f"\t{self._out_channels},\n"
            if SupportedNodes.conv in self.op_type:
                instance += f"\t{self._out_dim},\n"
                instance += f"\t{self._stride},\n"
                instance += f"\t{self._padding},\n"
        elif SupportedNodes.linear in self.op_type:
            instance += f"\t{self._in_features},\n"
            instance += f"\t{self._out_features},\n"
        else:
            raise ValueError(f"Expected [Conv/Linear], [Conv/Linear, BatchNormalization, Relu], [Conv/Linear, BatchNormalization] or [Conv/Linear, ReLu/ReLu6/Mul/Sigmoid], got {self.op_type} layer")
        instance += f"\t{self._simd},\n"
        instance += f"\t{self._pe},\n"
        instance += f"\t{self._weight_dtype},\n"
        instance += f"\tSlice<{self._input_dtype}>,\n" if self.dtype is not Dtypes.INT else f"\t{self._input_dtype},\n"
        instance += f"\tSlice<{self._output_dtype}>\n" if self.dtype is not Dtypes.INT else f"\t{self._output_dtype}\n"
        instance += "\t>\n"
        instance += f"\t({streams},\n"
        instance += f"\t{self._weights},\n"
        instance += f"\t{self._activations},\n"
        if self.dtype is Dtypes.INT:
            instance += f"\t{self._input_zero_point},\n"
        instance += "\tReps,\n"
        instance += "\tap_resource_dflt(),\n"
        instance += "\t0,\n"
        instance += f"\t&{self.name});\n"
        return instance

    def get_instance_split(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        instance += f"\t{self._input_accesses},\n"
        instance += f"\t{self._input_dtype}\n"
        instance += "\t>\n"
        instance += f"\t({streams},\n"
        instance += "\tReps,\n"
        instance += f"\t&{self.name});\n"
        return instance

    def get_instance_merge(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        instance += f"\t{self._simd},\n"
        instance += f"\t{self._input_dtype},\n"
        instance += f"\t{self._fifo_input_dtype},\n"
        instance += f"\t{self._output_dtype},\n"
        instance += f"\t{self._m0_dtype},\n"
        instance += "\tACT_MIN_VALUE,\n"
        instance += "\tACT_MAX_VALUE,\n"
        instance += f"\t{self._input_accesses}\n"
        instance += "\t>\n"
        instance += f"\t({streams},\n"
        instance += f"\t{self._input1_m0},\n"
        if not self.fixed_scale:
            instance += f"\t{self._input1_shift},\n"
        instance += f"\t{self._input1_zero_point},\n"
        instance += f"\t{self._input2_m0},\n"
        if not self.fixed_scale:
            instance += f"\t{self._input2_shift},\n"
        instance += f"\t{self._input2_zero_point},\n"
        instance += f"\t{self._output_zero_point},\n"
        instance += "\tReps,\n"
        instance += "\t0,\n"
        instance += f"\t&{self.name});\n"
        return instance

    def get_instance_upsample(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        instance += f"\t{self._out_dim},\n"
        instance += f"\t{self._in_dim},\n"
        instance += f"\t{self._in_channels},\n"
        instance += f"\t{self._input_dtype}\n"
        instance += "\t>\n"
        instance += f"\t({streams},\n"
        instance += "\tReps);\n"
        return instance

    def get_instance_maxpool(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        instance += f"\t{self._in_channels},\n"
        instance += f"\t{self._in_dim},\n"
        instance += f"\t{self._pe},\n"
        instance += f"\t{self._kernel_size},\n"
        instance += f"\t{self._stride},\n"
        instance += f"\tSlice<{self._input_dtype}>,\n"
        instance += f"\t{self._input_dtype}\n"
        instance += "\t>\n"
        instance += f"\t({streams},\n"
        instance += "\tReps);\n"
        return instance

    def get_instance_globalavgpool(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        instance += f"\t{self._in_channels},\n"
        instance += f"\t{self._pe},\n"
        instance += f"\t{self._in_dim},\n"
        instance += f"\t{self._out_dim},\n"
        instance += f"\t{self._input_dtype},\n"
        instance += f"\t{self._acc_dtype},\n"
        instance += f"\t{self._m0_dtype},\n"
        instance += "\tACT_MIN_VALUE,\n"
        instance += "\tACT_MAX_VALUE\n"
        instance += "\t>\n"
        instance += f"\t({streams},\n"
        instance += f"\t{self._m0},\n"
        if not self.fixed_scale:
            instance += f"\t{self._shift},\n"
        instance += f"\t{self._input_zero_point},\n"
        instance += f"\t{self._output_zero_point},\n"
        instance += "\tReps);\n"
        return instance

    def get_instance_expand(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        instance += f"\t{self._in_channels},\n"
        instance += f"\t{self._in_dim},\n"
        instance += f"\t{self._out_dim}\n"
        instance += "\t>\n"
        instance += f"\t({streams},\n"
        instance += "\tReps);\n"
        return instance

    def get_instance_clip(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        instance += f"\t{self._in_channels},\n"
        instance += f"\t{self._in_dim},\n"
        instance += f"\t{self._pe},\n"
        instance += f"\t{self._max},\n"
        instance += f"\t{self._min},\n"
        instance += f"\t{self._input_dtype}\n"
        instance += "\t>\n"
        instance += f"\t({streams},\n"
        instance += "\tReps);\n"
        return instance

    def get_instance_concat(self, streams):
        instance = ""
        instance += f"\t{self.layer_insatnce}\n"
        instance += "\t<\n"
        instance += f"\t{self._in_channels}/{self._simd},\n"
        instance += f"\t{self._in_dim}"
        if self.dtype is Dtypes.INT and self.fixed_scale:
            instance += ",\n"
            instance += f"\t{self._scale_dtype},\n"
            instance += f"\t{self._shift_dtype}"
        instance += "\n\t>\n"
        instance += f"\t({streams},\n"
        if self.dtype is Dtypes.INT and self.fixed_scale:
            for key in self._m_zp_dict:
                instance += f"\t{key},\n"
        instance += "\tReps);\n"
        return instance

    def get_instance_sdwc(self, streams):
        instance = ""
        streams = streams.replace("\n", " ")
        streams = streams.replace("\t", " ")
        instance += f"\t{self.layer_insatnce}<{self._input_stream_width}, {self._output_stream_width}, {self._input_accesses}>({streams}, Reps);\n"
        return instance

    def get_module_instance(self, input_stream_name, output_stream_name):

        conv_kernel_stride_padding = "Conv_Kernel_Stride_Padding_Batch"
        conv_kernel_stride = "Conv_Kernel_Stride_Batch"
        conv_padding = "Conv_Padding_Batch"
        conv = "Conv_Batch"
        conv_pw = "Conv_Pw_Batch"
        conv_dw = "Conv_Dw_Batch"
        conv_dw_padding = "Conv_Dw_Padding_Batch"
        conv_dw_kernel_stride_padding = "Conv_Dw_Kernel_Stride_Padding_Batch"
        tr_conv = "TransposeConv"
        linear = "Linear_Batch"
        merge = "AddStreams_Type_Batch"
        split2two = "DuplicateStreams_Batch"
        split2three = "TriplicateStreams_Batch"
        upsample = "UpsampleNearestNeighbour_Batch"
        maxpool = "Streaming_Maxpool_batch2d"
        globalavgpool = "Adaptive_Avg_Pool_2d"
        expand = "ExpandStream_Batch"
        clip = "ClipStream_Batch"
        concat = "ConcatStreams_Batch"
        sdwc = "StreamingDataWidthConverter_Batch"

        if SupportedNodes.conv in self.op_type:
            if (self.kernel_size > 1) and (self.kernel_size % self.stride != 0) and (self.padding > 0) and (self.group == 1):
                self.layer_insatnce = conv_kernel_stride_padding
            elif (self.kernel_size > 1) and (self.kernel_size % self.stride != 0) and (self.padding == 0) and (self.group == 1):
                self.layer_insatnce = conv_kernel_stride
            elif (self.kernel_size > 1) and (self.kernel_size % self.stride == 0) and (self.padding != 0) and (self.group == 1):
                self.layer_insatnce = conv_padding
            elif (self.kernel_size > 1) and (self.kernel_size % self.stride == 0) and (self.padding == 0) and (self.group == 1):
                self.layer_insatnce = conv
            elif (self.kernel_size == 1) and (self.stride == 1) and (self.padding == 0) and (self.group == 1):
                self.layer_insatnce = conv_pw
            elif (self.kernel_size % self.stride == 0) and (self.padding == 0) and (self.group == self.out_channels):
                self.layer_insatnce = conv_dw
            elif (self.kernel_size > 1) and (self.kernel_size % self.stride == 0) and (self.padding > 0) and (self.group == self.out_channels):
                self.layer_insatnce = conv_dw_padding
            elif (self.kernel_size > 1) and (self.kernel_size % self.stride != 0) and (self.padding > 0) and (self.group == self.out_channels):
                self.layer_insatnce = conv_dw_kernel_stride_padding
            else:
                raise NotImplementedError(f"This layer {self} is not supported yet. Kernel size: {self.kernel_size}, Stride: {self.stride}, Padding: {self.padding}, Group size: {self.group}")
        elif SupportedNodes.tr_conv in self.op_type:
            self.layer_insatnce = tr_conv
        elif SupportedNodes.linear in self.op_type:
            self.layer_insatnce = linear
        elif SupportedNodes.merge == self.op_type:
            self.layer_insatnce = merge
        elif SupportedNodes.split == self.op_type:
            if len(output_stream_name) == 2:
                self.layer_insatnce = split2two
            elif len(output_stream_name) == 3:
                self.layer_insatnce = split2three
            else:
                raise ValueError(f"This layer {self.op_type} has more outputs {len(output_stream_name)} than supported in hardware.")
        elif SupportedNodes.upsample == self.op_type:
            self.layer_insatnce = upsample
        elif SupportedNodes.maxpool == self.op_type:
            self.layer_insatnce = maxpool
        elif SupportedNodes.globalavgpool == self.op_type:
            self.layer_insatnce = globalavgpool
        elif SupportedNodes.expand == self.op_type:
            self.layer_insatnce = expand
        elif SupportedNodes.clip == self.op_type:
            self.layer_insatnce = clip
        elif SupportedNodes.concat == self.op_type:
            self.layer_insatnce = concat
            input_stream_name.reverse()  # reverse the input streams list to match the order of concatination in ONNX
        elif SupportedNodes.sdwc == self.op_type:
            self.layer_insatnce = sdwc
        else:
            raise NotImplementedError(f"This layer {self.op_type} is not supported yet.")

        streams = ""
        for i, stream in enumerate(input_stream_name):
            if i == 0:
                streams += stream + ",\n"
            else:
                streams += "\t" + stream + ",\n"
        for i, stream in enumerate(output_stream_name):
            if i == len(output_stream_name) - 1:
                streams += "\t" + stream
            else:
                streams += "\t" + stream + ",\n"

        if any(i in self.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]):
            return self.get_instance_layer(streams)
        elif SupportedNodes.split in self.op_type:
            return self.get_instance_split(streams)
        elif SupportedNodes.merge in self.op_type:
            return self.get_instance_merge(streams)
        elif SupportedNodes.upsample in self.op_type:
            return self.get_instance_upsample(streams)
        elif SupportedNodes.maxpool in self.op_type:
            return self.get_instance_maxpool(streams)
        elif SupportedNodes.globalavgpool in self.op_type:
            return self.get_instance_globalavgpool(streams)
        elif SupportedNodes.expand in self.op_type:
            return self.get_instance_expand(streams)
        elif SupportedNodes.clip in self.op_type:
            return self.get_instance_clip(streams)
        elif SupportedNodes.concat in self.op_type:
            return self.get_instance_concat(streams)
        elif SupportedNodes.sdwc in self.op_type:
            return self.get_instance_sdwc(streams)
        else:
            print(f"NodeHLs: get_module_instance: unsupported layer: {self.op_type}")
            return ""

    def get_num_params(self):
        num_params = 0

        if any(op in self.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
            num_params = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size / self.group
            if self.bias is not None:
                num_params += self.out_channels
            if any(i in self.op_type for i in [SupportedNodes.bn, SupportedNodes.ins]):
                num_params += 2 * self.out_channels
            if any(i in self.op_type for i in [SupportedNodes.mul]):
                num_params += 1
        elif SupportedNodes.linear in self.op_type:
            num_params = self.in_features * self.out_features
            if self.bias is not None:
                num_params += self.out_features
            if any(i in self.op_type for i in [SupportedNodes.bn, SupportedNodes.ins]):
                num_params += 2 * self.out_features
            if any(i in self.op_type for i in [SupportedNodes.mul]):
                num_params += 1
        else:
            num_params = 0

        return int(num_params)

    def get_latency_cc_node(self, simd=None, pe=None):
        '''
        This function returns the latency of the node as a tuple,
            The first element is the latency of buffer if it exists and the second element is the latency of the MAC unit
        '''
        # For debugging purposes
        simd = self.simd if simd is None else simd
        pe = self.pe if pe is None else pe

        latency = 0
        if SupportedNodes.conv in self.op_type:
            if self.kernel_size == 1 and self.stride == 1:  # This covers Conv_Pw_Batch
                buffer_latency = self.in_dim * self.in_dim * self.in_channels / simd
            else:
                multiplying_factor = self.in_channels / simd
                cycles_write_block = self.out_dim * self.kernel_size * self.kernel_size * multiplying_factor
                cycles_read_block = self.stride * self.in_dim * multiplying_factor
                max_cycles = max(cycles_write_block, cycles_read_block)
                if self.kernel_size % self.stride == 0:  # This covers Conv_Batch and Conv_Dw_Batch
                    buffer_latency = self.in_dim * self.kernel_size * multiplying_factor + self.out_dim * max_cycles
                else:  # This covers Conv_Kernel_Stride_Padding_Batch and Conv_Dw_Kernel_Stride_Padding_Batch
                    buffer_latency = (self.in_dim * self.kernel_size * multiplying_factor) + (self.out_dim - 1) * max_cycles + max(cycles_write_block, self.out_dim)
            if self.group == self.in_channels:
                nf = self.in_channels / pe
                sf = self.kernel_size * self.kernel_size
                reps = self.out_dim * self.out_dim
            else:
                nf = self.out_channels / pe
                sf = (self.kernel_size * self.kernel_size * self.in_channels) / simd
                reps = self.out_dim * self.out_dim
            mvau_latency = nf * sf * reps
            return math.ceil(buffer_latency), math.ceil(mvau_latency)
        elif SupportedNodes.tr_conv in self.op_type:
            in_fold = self.in_channels / simd
            buffer_latency_first_row = self.in_dim * in_fold
            total_write_cycles = self.out_dim * self.out_dim * in_fold
            buffer_latency = buffer_latency_first_row + total_write_cycles
            nf = self.out_channels / pe
            sf = self.in_channels / simd
            reps = self.out_dim * self.out_dim
            mvau_latency = nf * sf * reps
            return math.ceil(buffer_latency), math.ceil(mvau_latency)
        elif SupportedNodes.linear in self.op_type:
            latency = 0, math.ceil((self.in_features / simd) * (self.out_features / pe))
        elif SupportedNodes.split in self.op_type:
            latency = 0, 0
        elif SupportedNodes.merge in self.op_type:
            latency = 0, 0
        elif SupportedNodes.upsample in self.op_type:
            latency = 0, 0
        elif SupportedNodes.maxpool in self.op_type:
            latency = 0, 0
        else:
            latency = 0, 0
            #raise Exception(f"Unsupported layer: {self.op_type}")
        return latency

    def get_depth_cc_node(self):
        if SupportedNodes.conv in self.op_type:
            if self.in_channels % self.simd != 0:
                raise ValueError(f"Number of input channels: {self.in_channels} has to be a multiple of input parallelism: {self.simd}")
            if self.out_channels % self.pe != 0:
                raise ValueError(f"Number of output channels: {self.out_channels} has to be a multiple of output parallelism: {self.pe}")
        elif SupportedNodes.linear in self.op_type:
            if self.in_features % self.simd != 0:
                raise ValueError(f"Number of input channels: {self.in_features} has to be a multiple of input parallelism: {self.simd}")
            if self.out_features % self.pe != 0:
                raise ValueError(f"Number of output channels: {self.out_features} has to be a multiple of output parallelism: {self.pe}")

        depth = 0
        if SupportedNodes.conv in self.op_type:
            if self.kernel_size == 1:  # This covers Conv_Pw_Batch
                conv_input_generator_depth = 4
                dot_product_depth = self.in_channels / self.simd
                depth = conv_input_generator_depth + dot_product_depth
            elif self.group == 1:  # This covers Conv_Kernel_Stride_Padding_Batch, Conv_Kernel_Stride_Batch, Conv_Padding_Batch, Conv_Batch
                conv_input_generator_depth = self.in_dim * self.kernel_size * (self.in_channels / self.simd)
                dot_product_depth = self.in_channels / self.simd
                depth = conv_input_generator_depth + dot_product_depth
            else: # This covers Conv_Dw_Batch, Conv_Dw_Padding_Batch, Conv_Dw_Kernel_Stride_Padding_Batch
                conv_input_generator_depth = self.in_dim * self.kernel_size * (self.in_channels / self.simd)
                dot_product_depth = self.pe
                depth = conv_input_generator_depth + dot_product_depth
            if SupportedNodes.bn in self.op_type:
                depth += 4
            if SupportedNodes.ins in self.op_type:
                depth += 4
            if SupportedNodes.relu in self.op_type:
                depth += 4
            if SupportedNodes.relu6 in self.op_type:
                depth += 4
        elif SupportedNodes.linear in self.op_type:
            depth = self.in_features / self.simd
            if SupportedNodes.bn in self.op_type:
                depth += 4
            if SupportedNodes.ins in self.op_type:
                depth += 4
            if SupportedNodes.relu in self.op_type:
                depth += 4
            if SupportedNodes.relu6 in self.op_type:
                depth += 4
        elif SupportedNodes.tr_conv in self.op_type:
            tr_conv_input_generator_depth = self.in_dim * (self.in_channels / self.simd)
            dot_product_depth = self.in_channels / self.simd
            depth = tr_conv_input_generator_depth + dot_product_depth
        elif SupportedNodes.split in self.op_type:
            depth = 4
        elif SupportedNodes.merge in self.op_type:
            depth = 4
        elif SupportedNodes.upsample in self.op_type:
            depth = 4
        elif SupportedNodes.maxpool in self.op_type:
            depth = 4
        else:
            depth = 0
            #raise Exception(f"Unsupported layer: {self.op_type}")
        return int(depth)

    def get_macros_merge(self):
        assert SupportedNodes.merge in self.op_type, f"Expected Merge layer, got {self.op_type}"
        macros = ""
        macros += f"#define {self._simd:70}{self.simd}\n"
        macros += f"#define {self._pe:70}{self.pe}\n"

        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        self.in_bits_aligned = shift_bit_length(self.in_bits)
        self.input_stream_width = int(self.in_bits * self.simd)
        self.input_mem_width = int(self.in_bits_aligned * self.simd)
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_bits_aligned:70}{self.in_bits_aligned}\n"
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"
        macros += f"#define {self._input_mem_width:70}{self.input_mem_width}\n"

        macros += f"#define {self._fifo_in_bits:70}{self.fifo_in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._fifo_in_int_bits:70}{self.fifo_in_int_bits}\n"
        self.fifo_in_bits_aligned = shift_bit_length(self.fifo_in_bits)
        self.fifo_input_stream_width = int(self.fifo_in_bits * self.simd)
        self.fifo_input_mem_width = int(self.fifo_in_bits_aligned * self.simd)
        macros += f"#define {self._fifo_in_bits_aligned:70}{self.fifo_in_bits_aligned}\n"
        macros += f"#define {self._fifo_input_stream_width:70}{self.fifo_input_stream_width}\n"
        macros += f"#define {self._fifo_input_mem_width:70}{self.fifo_input_mem_width}\n"

        macros += f"#define {self._input_accesses:70}{self.input_accesses}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.out_bits_aligned = shift_bit_length(self.out_bits)
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)
        macros += f"#define {self._out_bits_aligned:70}{self.out_bits_aligned}\n"
        macros += f"#define {self._output_stream_width:70}{self.output_stream_width}\n"
        macros += f"#define {self._output_mem_width:70}{self.output_mem_width}\n"
        macros += f"#define {self._output_accesses:70}{self.output_accesses}\n"
        if self.dtype is Dtypes.INT:
            macros += f"#define {self._input1_m0:70}{self.input1_m0}\n"
            if not self.fixed_scale:
                macros += f"#define {self._input1_shift:70}{self.input1_shift}\n"
            macros += f"#define {self._input1_zero_point:70}{self.input1_zero_point}\n"
            macros += f"#define {self._input2_m0:70}{self.input2_m0}\n"
            if not self.fixed_scale:
                macros += f"#define {self._input2_shift:70}{self.input2_shift}\n"
            macros += f"#define {self._input2_zero_point:70}{self.input2_zero_point}\n"
            macros += f"#define {self._output_zero_point:70}{self.output_zero_point}\n"
            if self.fixed_scale:
                macros += f"#define {self._m0_bits:70}{self.m0_bits}\n"
                macros += f"#define {self._m0_int_bits:70}{self.m0_int_bits}\n"

        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._fifo_in_bits},{self._fifo_in_int_bits},AP_RND_ZERO,AP_WRAP> {self._fifo_input_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._fifo_input_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
        else:
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._fifo_in_bits}> {self._fifo_input_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"
            if self.fixed_scale:
                macros += f"typedef ap_ufixed<{self._m0_bits},{self._m0_int_bits}> {self._m0_dtype};\n"


        macros += "\n"
        return macros

    def get_macros_split(self):
        assert SupportedNodes.split in self.op_type, f"Expected Split layer, got {self.op_type}"
        macros = ""
        macros += f"#define {self._simd:70}{self.simd}\n"
        macros += f"#define {self._pe:70}{self.pe}\n"

        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        self.in_bits_aligned = shift_bit_length(self.in_bits)
        self.input_stream_width = int(self.in_bits * self.simd)
        self.input_mem_width = int(self.in_bits_aligned * self.simd)
        macros += f"#define {self._in_bits_aligned:70}{self.in_bits_aligned}\n"
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"
        macros += f"#define {self._input_mem_width:70}{self.input_mem_width}\n"
        macros += f"#define {self._input_accesses:70}{self.input_accesses}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.out_bits_aligned = shift_bit_length(self.out_bits)
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)
        macros += f"#define {self._out_bits_aligned:70}{self.out_bits_aligned}\n"
        macros += f"#define {self._output_stream_width:70}{self.output_stream_width}\n"
        macros += f"#define {self._output_mem_width:70}{self.output_mem_width}\n"
        macros += f"#define {self._output_accesses:70}{self.output_accesses}\n"

        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
        else:
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"

        macros += "\n"
        return macros

    def get_macros_upsample(self):
        assert SupportedNodes.upsample in self.op_type, f"Expected UpSample layer, got {self.op_type}"
        macros = ""
        macros += f"#define {self._simd:70}{self.simd}\n"
        macros += f"#define {self._pe:70}{self.pe}\n"

        macros += f"#define {self._in_channels:70}{self.in_channels}\n"
        macros += f"#define {self._in_dim:70}{self.in_dim}\n"
        macros += f"#define {self._out_channels:70}{self.out_channels}\n"
        macros += f"#define {self._out_dim:70}{self.out_dim}\n"
        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        self.in_bits_aligned = shift_bit_length(self.in_bits)
        self.input_stream_width = int(self.in_bits * self.simd)
        self.input_mem_width = int(self.in_bits_aligned * self.simd)
        macros += f"#define {self._in_bits_aligned:70}{self.in_bits_aligned}\n"
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"
        macros += f"#define {self._input_mem_width:70}{self.input_mem_width}\n"
        macros += f"#define {self._input_accesses:70}{self.input_accesses}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.out_bits_aligned = shift_bit_length(self.out_bits)
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)
        macros += f"#define {self._out_bits_aligned:70}{self.out_bits_aligned}\n"
        macros += f"#define {self._output_stream_width:70}{self.output_stream_width}\n"
        macros += f"#define {self._output_mem_width:70}{self.output_mem_width}\n"
        macros += f"#define {self._output_accesses:70}{self.output_accesses}\n"

        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
        else:
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"

        macros += "\n"
        return macros

    def get_macros_maxpool(self):
        assert SupportedNodes.maxpool in self.op_type, f"Expected MaxPool layer, got {self.op_type}"
        macros = ""
        macros += f"#define {self._simd:70}{self.simd}\n"
        macros += f"#define {self._pe:70}{self.pe}\n"

        macros += f"#define {self._kernel_size:70}{self.kernel_size}\n"
        macros += f"#define {self._stride:70}{self.stride}\n"
        macros += f"#define {self._in_channels:70}{self.in_channels}\n"
        macros += f"#define {self._in_dim:70}{self.in_dim}\n"
        macros += f"#define {self._out_channels:70}{self.out_channels}\n"
        macros += f"#define {self._out_dim:70}{self.out_dim}\n"
        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        self.in_bits_aligned = shift_bit_length(self.in_bits)
        self.input_stream_width = int(self.in_bits * self.simd)
        self.input_mem_width = int(self.in_bits_aligned * self.simd)
        macros += f"#define {self._in_bits_aligned:70}{self.in_bits_aligned}\n"
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"
        macros += f"#define {self._input_mem_width:70}{self.input_mem_width}\n"
        macros += f"#define {self._input_accesses:70}{self.input_accesses}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.out_bits_aligned = shift_bit_length(self.out_bits)
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)
        macros += f"#define {self._out_bits_aligned:70}{self.out_bits_aligned}\n"
        macros += f"#define {self._output_stream_width:70}{self.output_stream_width}\n"
        macros += f"#define {self._output_mem_width:70}{self.output_mem_width}\n"
        macros += f"#define {self._output_accesses:70}{self.output_accesses}\n"

        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
        else:
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"

        macros += "\n"
        return macros

    def get_macros_layer(self):
        macros = ""
        if SupportedNodes.conv in self.op_type or SupportedNodes.tr_conv in self.op_type:
            macros += f"#define {self._kernel_size:70}{self.kernel_size}\n"
            macros += f"#define {self._in_channels:70}{self.in_channels}\n"
            macros += f"#define {self._in_dim:70}{self.in_dim}\n"
            macros += f"#define {self._out_channels:70}{self.out_channels}\n"
            macros += f"#define {self._out_dim:70}{self.out_dim}\n"
            macros += f"#define {self._group:70}{self.group}\n"
            macros += f"#define {self._stride:70}{self.stride}\n"
            macros += f"#define {self._padding:70}{self.padding}\n"
            activation_tiles = int(self.out_channels / self.pe)
            weight_tiles = int(self.kernel_size * self.kernel_size * self.in_channels * self.out_channels / self.group / self.pe / self.simd)
        elif SupportedNodes.linear in self.op_type:
            macros += f"#define {self._in_features:70}{self.in_features}\n"
            macros += f"#define {self._out_features:70}{self.out_features}\n"
            activation_tiles = int(self.out_features / self.pe)
            weight_tiles = int(self.in_features * self.out_features / self.pe / self.simd)
        else:
            raise ValueError(f"Expected [Conv/Linear], [Conv/Linear, BatchNormalization, Relu] or [Conv/Linear, BatchNormalization], got {self.op_type} layer")
        macros += f"#define {self._simd:70}{self.simd}\n"
        macros += f"#define {self._pe:70}{self.pe}\n"

        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        self.in_bits_aligned = shift_bit_length(self.in_bits)
        self.input_stream_width = int(self.in_bits * self.simd)
        self.input_mem_width = int(self.in_bits_aligned * self.simd)
        macros += f"#define {self._in_bits_aligned:70}{self.in_bits_aligned}\n"
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"
        macros += f"#define {self._input_mem_width:70}{self.input_mem_width}\n"
        macros += f"#define {self._input_accesses:70}{self.input_accesses}\n"
        macros += f"#define {self._weight_bits:70}{self.weight_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._weight_int_bits:70}{self.weight_int_bits}\n"
        weight_simd_width = int(self.weight_bits * self.simd)
        weight_simd_pe_width = int(self.weight_bits * self.simd * self.pe)
        macros += f"#define {self._weight_simd_width:70}{weight_simd_width}\n"
        macros += f"#define {self._weight_simd_pe_width:70}{weight_simd_pe_width}\n"
        if self.bias is not None:
            macros += f"#define {self._bias_bits:70}{self.bias_bits}\n"
            if self.dtype is not Dtypes.INT:
                macros += f"#define {self._bias_int_bits:70}{self.bias_int_bits}\n"
        macros += f"#define {self._acc_bits:70}{self.acc_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._acc_int_bits:70}{self.acc_int_bits}\n"
        macros += f"#define {self._activation_tiles:70}{activation_tiles}\n"
        macros += f"#define {self._weight_tiles:70}{weight_tiles}\n"
        if any(i in self.op_type for i in [SupportedNodes.bn, SupportedNodes.ins]):
            macros += f"#define {self._scale_bits:70}{self.scale_bits}\n"
            macros += f"#define {self._scale_int_bits:70}{self.scale_int_bits}\n"
            macros += f"#define {self._shift_bits:70}{self.shift_bits}\n"
            macros += f"#define {self._shift_int_bits:70}{self.shift_int_bits}\n"
            macros += f"#define {self._norm_acc_bits:70}{self.norm_acc_bits}\n"
            macros += f"#define {self._norm_acc_int_bits:70}{self.norm_acc_int_bits}\n"
        if SupportedNodes.mul in self.op_type:
            macros += f"#define {self._scale_bits:70}{self.scale_bits}\n"
            if self.dtype is not Dtypes.INT:
                macros += f"#define {self._scale_int_bits:70}{self.scale_int_bits}\n"
            macros += f"#define {self._act_acc_bits:70}{self.act_acc_bits}\n"
            if self.dtype is not Dtypes.INT:
                macros += f"#define {self._act_acc_int_bits:70}{self.act_acc_int_bits}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.out_bits_aligned = shift_bit_length(self.out_bits)
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)
        macros += f"#define {self._out_bits_aligned:70}{self.out_bits_aligned}\n"
        macros += f"#define {self._output_stream_width:70}{self.output_stream_width}\n"
        macros += f"#define {self._output_mem_width:70}{self.output_mem_width}\n"
        macros += f"#define {self._output_accesses:70}{self.output_accesses}\n"
        if self.dtype is Dtypes.INT:
            macros += f"#define {self._m0:70}{self.m0}\n"
            if self.fixed_scale:
                macros += f"#define {self._m0_bits:70}{self.m0_bits}\n"
                macros += f"#define {self._m0_int_bits:70}{self.m0_int_bits}\n"
            else:
                macros += f"#define {self._shift:70}{self.shift}\n"

            macros += f"#define {self._input_zero_point:70}{self.input_zero_point}\n"
            macros += f"#define {self._output_zero_point:70}{self.output_zero_point}\n"
        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._weight_bits},{self._weight_int_bits},AP_RND_ZERO,AP_WRAP> {self._weight_dtype};\n"
            if self.bias is not None:
                macros += f"typedef ap_fixed<{self._bias_bits},{self._bias_int_bits},AP_RND_ZERO,AP_WRAP> {self._bias_dtype};\n"
            macros += f"typedef ap_fixed<{self._acc_bits},{self._acc_int_bits},AP_RND_ZERO,AP_WRAP> {self._acc_dtype};\n"
            if any(i in self.op_type for i in [SupportedNodes.bn, SupportedNodes.ins]):
                macros += f"typedef ap_fixed<{self._scale_bits},{self._scale_int_bits},AP_RND_ZERO,AP_WRAP> {self._scale_dtype};\n"
                macros += f"typedef ap_fixed<{self._shift_bits},{self._shift_int_bits},AP_RND_ZERO,AP_WRAP> {self._shift_dtype};\n"
                macros += f"typedef ap_fixed<{self._norm_acc_bits},{self._norm_acc_int_bits},AP_RND_ZERO,AP_WRAP> {self._norm_acc_dtype};\n"
            if SupportedNodes.mul in self.op_type:
                macros += f"typedef ap_fixed<{self._scale_bits},{self._scale_int_bits},AP_RND_ZERO,AP_WRAP> {self._scale_dtype};\n"
                macros += f"typedef ap_fixed<{self._act_acc_bits},{self._act_acc_int_bits},AP_RND_ZERO,AP_WRAP> {self._act_acc_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._weight_dtype};\n"
            if self.bias is not None:
                macros += f"typedef float {self._bias_dtype};\n"
            macros += f"typedef float {self._acc_dtype};\n"
            if any(i in self.op_type for i in [SupportedNodes.bn, SupportedNodes.ins]):
                macros += f"typedef float {self._scale_dtype};\n"
                macros += f"typedef float {self._shift_dtype};\n"
                macros += f"typedef float {self._norm_acc_dtype};\n"
            if SupportedNodes.mul in self.op_type:
                macros += f"typedef float {self._scale_dtype};\n"
                macros += f"typedef float {self._act_acc_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
        else:
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._weight_bits}> {self._weight_dtype};\n"
            if self.bias is not None:
                macros += f"typedef ap_int<{self._bias_bits}> {self._bias_dtype};\n"
            macros += f"typedef ap_int<{self._acc_bits}> {self._acc_dtype};\n"
            if any(i in self.op_type for i in [SupportedNodes.bn, SupportedNodes.ins]):
                raise NotImplementedError("Not implemented")
            if SupportedNodes.mul in self.op_type:
                macros += f"typedef ap_int<{self._scale_bits}> {self._scale_dtype};\n"
                macros += f"typedef ap_int<{self._act_acc_bits}> {self._act_acc_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"
            if self.fixed_scale:
                macros+= f"typedef ap_ufixed<{self._m0_bits}, {self._m0_int_bits}, AP_RND_ZERO, AP_WRAP> {self._m0_dtype};\n"
            else:
                macros+= f"typedef ap_uint<32> {self._m0_dtype};\n"
        macros += "\n"
        return macros

    def get_macros_globalavgpool(self):
        assert SupportedNodes.globalavgpool in self.op_type, f"Expected GlobalAvgBool2d layer, got {self.op_type}"
        macros = ""
        macros += f"#define {self._simd:70}{self.simd}\n"
        macros += f"#define {self._pe:70}{self.pe}\n"

        macros += f"#define {self._in_channels:70}{self.in_channels}\n"
        macros += f"#define {self._in_dim:70}{self.in_dim}\n"
        macros += f"#define {self._out_dim:70}{self.out_dim}\n"
        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        self.in_bits_aligned = shift_bit_length(self.in_bits)
        self.input_stream_width = int(self.in_bits * self.simd)
        self.input_mem_width = int(self.in_bits_aligned * self.simd)
        macros += f"#define {self._in_bits_aligned:70}{self.in_bits_aligned}\n"
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"
        macros += f"#define {self._input_mem_width:70}{self.input_mem_width}\n"
        macros += f"#define {self._input_accesses:70}{self.input_accesses}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.out_bits_aligned = shift_bit_length(self.out_bits)
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)
        macros += f"#define {self._out_bits_aligned:70}{self.out_bits_aligned}\n"
        macros += f"#define {self._output_stream_width:70}{self.output_stream_width}\n"
        macros += f"#define {self._output_mem_width:70}{self.output_mem_width}\n"
        macros += f"#define {self._output_accesses:70}{self.output_accesses}\n"
        macros += f"#define {self._acc_bits:70}{self.acc_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._acc_int_bits:70}{self.acc_int_bits}\n"
        else:
            macros += f"#define {self._m0:70}{self.m0}\n"
            if not self.fixed_scale:
                macros += f"#define {self._shift:70}{self.shift}\n"
            else:
                macros += f"#define {self._m0_bits:70}{self.m0_bits}\n"
                macros += f"#define {self._m0_int_bits:70}{self.m0_int_bits}\n"
            macros += f"#define {self._input_zero_point:70}{self.input_zero_point}\n"
            macros += f"#define {self._output_zero_point:70}{self.output_zero_point}\n"

        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
            macros += f"typedef ap_fixed<{self._acc_bits},{self._acc_int_bits},AP_RND_ZERO,AP_WRAP> {self._acc_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
            macros += f"typedef float {self._acc_dtype};\n"
        else:
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"
            macros += f"typedef ap_int<{self._acc_bits}> {self._acc_dtype};\n"
            if self.fixed_scale:
                macros += f"typedef ap_ufixed<{self._m0_bits}, {self._m0_int_bits}, AP_RND_ZERO, AP_WRAP> {self._m0_dtype};\n"
        macros += "\n"
        return macros

    def get_macros_expand(self):
        assert SupportedNodes.expand in self.op_type, f"Expected Expand layer, got {self.op_type}"
        macros = ""

        macros += f"#define {self._in_channels:70}{self.in_channels}\n"
        macros += f"#define {self._in_dim:70}{self.in_dim}\n"
        macros += f"#define {self._out_dim:70}{self.out_dim}\n"
        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.input_stream_width = int(self.in_bits * self.simd)
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"


        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
        else:
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"

        macros += "\n"
        return macros

    def get_macros_clip(self):
        assert SupportedNodes.clip in self.op_type, f"Expected Clip layer, got {self.op_type}"
        macros = ""

        macros += f"#define {self._in_channels:70}{self.in_channels}\n"
        macros += f"#define {self._in_dim:70}{self.in_dim}\n"
        macros += f"#define {self._pe:70}{self.pe}\n"
        macros += f"#define {self._max:70}{self.max}\n"
        macros += f"#define {self._min:70}{self.min}\n"
        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.input_stream_width = int(self.in_bits * self.simd)
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"

        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
        else:
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"

        macros += "\n"
        return macros

    def get_macros_concat(self):
        assert SupportedNodes.concat in self.op_type, f"Expected Concat layer, got {self.op_type}"
        macros = ""
        macros += f"#define {self._simd:70}{self.simd}\n"
        macros += f"#define {self._pe:70}{self.pe}\n"
        macros += f"#define {self._in_channels:70}{self.in_channels}\n"
        macros += f"#define {self._in_dim:70}{self.in_dim}\n"
        macros += f"#define {self._in_bits:70}{self.in_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._in_int_bits:70}{self.in_int_bits}\n"
        macros += f"#define {self._out_bits:70}{self.out_bits}\n"
        if self.dtype is not Dtypes.INT:
            macros += f"#define {self._out_int_bits:70}{self.out_int_bits}\n"
        self.input_stream_width = int(self.in_bits * self.simd)
        macros += f"#define {self._input_stream_width:70}{self.input_stream_width}\n"
        self.output_stream_width = int(self.out_bits * self.pe)
        self.output_mem_width = int(self.out_bits_aligned * self.pe)
        macros += f"#define {self._out_bits_aligned:70}{self.out_bits_aligned}\n"
        macros += f"#define {self._output_stream_width:70}{self.output_stream_width}\n"
        macros += f"#define {self._output_accesses:70}{self.output_accesses}\n"
        if self.dtype is Dtypes.FIXED:
            macros += f"typedef ap_fixed<{self._in_bits},{self._in_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
            macros += f"typedef ap_fixed<{self._out_bits},{self._out_int_bits},AP_RND_ZERO,AP_WRAP> {self._output_dtype};\n"
        elif self.dtype is Dtypes.FLOAT:
            macros += f"typedef float {self._input_dtype};\n"
            macros += f"typedef float {self._output_dtype};\n"
        else:
            if self.fixed_scale:
                for k,v in self._m_zp_dict.items():
                    macros += f"#define {k:70}{v}\n"
                macros += f"#define {self._scale_bits:70}{self.scale_bits}\n"
                macros += f"#define {self._scale_int_bits:70}{self.scale_int_bits}\n"
                macros += f"#define {self._shift_bits:70}{self.shift_bits}\n"
                macros += f"#define {self._shift_int_bits:70}{self.shift_int_bits}\n"
                macros += f"typedef ap_ufixed<{self._scale_bits},{self._scale_int_bits},AP_RND_ZERO,AP_WRAP> {self._scale_dtype};\n"
                macros += f"typedef ap_fixed<{self._shift_bits},{self._shift_int_bits},AP_RND_ZERO,AP_WRAP> {self._shift_dtype};\n"
            macros += f"typedef ap_int<{self._in_bits}> {self._input_dtype};\n"
            macros += f"typedef ap_int<{self._out_bits}> {self._output_dtype};\n"
        macros += "\n"
        return macros

    def get_macros(self):
        if any(i in self.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]):
            return self.get_macros_layer()
        elif SupportedNodes.split in self.op_type:
            return self.get_macros_split()
        elif SupportedNodes.merge in self.op_type:
            return self.get_macros_merge()
        elif SupportedNodes.upsample in self.op_type:
            return self.get_macros_upsample()
        elif SupportedNodes.maxpool in self.op_type:
            return self.get_macros_maxpool()
        elif SupportedNodes.globalavgpool in self.op_type:
            return self.get_macros_globalavgpool()
        elif SupportedNodes.expand in self.op_type:
            return self.get_macros_expand()
        elif SupportedNodes.clip in self.op_type:
            return self.get_macros_clip()
        elif SupportedNodes.concat in self.op_type:
            return self.get_macros_concat()
        elif SupportedNodes.sdwc in self.op_type:
            return ""
        else:
            print(f"NodeHLs: get_macros: unsupported layer: {self.op_type}")
            return ""

    def get_params(self):
        params = ""
        if any(i in self.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]):
            weight_quantizer = WeightQuantizationScheme(q_type='fixed_signed', bit_width=self.weight_bits, int_bit_width=self.weight_int_bits)
            if SupportedNodes.conv in self.op_type:
                weight = weight_quantizer(torch.from_numpy(self.weight)) if self.dtype is Dtypes.FIXED else torch.from_numpy(self.weight)
                weight = conv_weight_mapper_2D(weight, self.pe, self.simd, self.weight_bits, self.weight_int_bits, self.dtype is not Dtypes.FIXED)
            elif SupportedNodes.tr_conv in self.op_type:
                weight = weight_quantizer(torch.from_numpy(self.weight)) if self.dtype is Dtypes.FIXED else torch.from_numpy(self.weight)
                weight = tr_conv_weight_mapper_2D(weight, self.pe, self.simd, self.weight_bits, self.weight_int_bits, self.dtype is not Dtypes.FIXED)
            elif SupportedNodes.linear in self.op_type:
                weight = weight_quantizer(torch.from_numpy(self.weight)) if self.dtype is Dtypes.FIXED else torch.from_numpy(self.weight)
                weight = fc_weight_mapper_2D(weight, self.pe, self.simd, self.weight_bits, self.weight_int_bits, self.dtype is not Dtypes.FIXED)
            else:
                raise ValueError("Unexpected operation type. Did you just invent a new layer?")
            if self.dtype is Dtypes.FIXED:
                params += f"static ap_uint<{self._weight_simd_width}> {self._weights}[{self._pe}][{self._weight_tiles}] = {{\n{weight}}};\n"
            elif self.dtype is Dtypes.FLOAT:
                params += f"static float {self._weights}[{self._pe}][{self._weight_tiles}] = {{\n{weight}}};\n"
            else:
                params += f"static ap_int<{self._weight_bits}> {self._weights}[{self._pe}][{self._weight_tiles}][{self._simd}] = {{\n{weight}}};\n"

            if self.bias is not None:
                bias_quantizer = BiasQuantizationScheme(q_type='fixed_signed', bit_width=self.bias_bits, int_bit_width=self.bias_int_bits)
                bias = torch.from_numpy(np.copy(self.bias))
                bias = bias_quantizer(bias) if self.dtype is Dtypes.FIXED else bias
                bias = conv_bias_mapper_2D(bias, self.pe, self.bias_bits, self.bias_int_bits, self.dtype)
                if SupportedNodes.relu in self.op_type:
                    params += f"static BiasReLuActivation<{self._pe}, {self._activation_tiles}, {self._bias_dtype}, {self._acc_dtype}, {self._output_dtype}"
                    if self.dtype is Dtypes.INT:
                        params += f", {self._m0_dtype}> {self._activations} = {{\n.m_bias = {{{bias}}},\n.m0 = {self._m0},"
                        if not self.fixed_scale:
                            params += f"\n.sh_val = {self._shift},"
                        params += f"\n.zero_point = {self._output_zero_point}}};\n"
                    else:
                        params += f"> {self._activations} = {{\n.m_bias = {{{bias}}}}};\n"
                elif SupportedNodes.relu6 in self.op_type:
                    params += f"static BiasReLu6Activation<{self._pe}, {self._activation_tiles}, {self._bias_dtype}, {self._acc_dtype}, {self._output_dtype}"
                    if self.dtype is Dtypes.INT:
                        params += f", {self._m0_dtype}> {self._activations} = {{\n.m_bias = {{{bias}}},\n.m0 = {self._m0},"
                        if not self.fixed_scale:
                            params += f"\n.sh_val = {self._shift},"
                        params += f"\n.zero_point = {self._output_zero_point}}};\n"
                    else:
                        params += f"> {self._activations} = {{\n.m_bias = {{{bias}}}}};\n"
                elif SupportedNodes.mul in self.op_type:
                    scale_quantizer = BiasQuantizationScheme(q_type='fixed_signed', bit_width=self.scale_bits, int_bit_width=self.scale_int_bits)
                    scale = torch.from_numpy(self.scale)
                    scale = scale_quantizer(scale) if self.dtype is Dtypes.FIXED else scale
                    scale = scale.item()
                    params += "static BiasMulActivation<{:},{:},{:},{:},{:},{:},{:}> {:} = {{\n.m_bias = {{{:}}},\n.m_scale = {:}}};\n".format(self._pe, self._activation_tiles, self._bias_dtype, self._acc_dtype, self._scale_dtype, self._act_acc_dtype, self._output_dtype, self._activations, bias, scale)
                elif SupportedNodes.sigmoid in self.op_type:
                    params += "static BiasSigmoidActivation<{:},{:},{:},{:},{:}> {:} = {{\n.m_bias = {:}}};\n".format(self._pe, self._activation_tiles, self._bias_dtype, self._acc_dtype, self._output_dtype, self._activations, bias)
                elif SupportedNodes.hardswish in self.op_type:
                    params += "static BiasHardSwichActivation<{:},{:},{:},{:},{:}> {:} = {{\n.m_bias = {:}}};\n".format(self._pe, self._activation_tiles, self._bias_dtype, self._acc_dtype, self._output_dtype, self._activations, bias)
                else:
                    params += f"static BiasActivation<{self._pe}, {self._activation_tiles}, {self._bias_dtype}, {self._acc_dtype}, {self._output_dtype}"
                    if self.dtype is Dtypes.INT:
                        params += f", {self._m0_dtype}> {self._activations} = {{\n.m_bias = {{{bias}}},\n.m0 = {self._m0},"
                        if not self.fixed_scale:
                            params += f"\n.sh_val = {self._shift},"
                        params += f"\n.zero_point = {self._output_zero_point},\n.min_val = ACT_MIN_VALUE,\n.max_val = ACT_MAX_VALUE}};\n"
                    else:
                        params += f"> {self._activations} = {{\n.m_bias = {{{bias}}}}};\n"
            elif SupportedNodes.mul in self.op_type:
                scale_quantizer = BiasQuantizationScheme(q_type='fixed_signed', bit_width=self.scale_bits, int_bit_width=self.scale_int_bits)
                scale = scale_quantizer(torch.from_numpy(self.scale)) if self.dtype is Dtypes.FIXED else torch.from_numpy(self.scale)
                scale = scale.item()
                params += "static MulActivation<{:},{:},{:},{:},{:},{:}> {:} = {{{:}}};\n".format(self._pe, self._activation_tiles, self._acc_dtype, self._scale_dtype, self._act_acc_dtype, self._output_dtype, self._activations, scale)
            elif SupportedNodes.bn in self.op_type:
                scale_quantizer = WeightQuantizationScheme(q_type='fixed_signed', bit_width=self.scale_bits, int_bit_width=self.scale_int_bits)
                shift_quantizer = WeightQuantizationScheme(q_type='fixed_signed', bit_width=self.shift_bits, int_bit_width=self.shift_int_bits)
                scale = torch.from_numpy(self.norm_weight) / torch.sqrt(torch.from_numpy(self.running_var) + self.epsilon)
                scale = scale_quantizer(scale) if self.dtype is Dtypes.FIXED else scale
                shift = torch.from_numpy(self.running_mean) * scale - torch.from_numpy(self.norm_bias)
                shift = shift_quantizer(shift) if self.dtype is Dtypes.FIXED else shift
                scale = conv_bias_mapper_2D(scale, self.pe, self.scale_bits, self.scale_int_bits)
                shift = conv_bias_mapper_2D(shift, self.pe, self.shift_bits, self.shift_int_bits)
                if SupportedNodes.relu in self.op_type:
                    params += "static BatchNormReLuActivation<{:},{:},{:},{:},{:},{:},{:}> {:} = {{\n.m_scale = {{{:}}},\n.m_shift = {{{:}}}}};\n".format(self._pe, self._activation_tiles, self._acc_dtype, self._scale_dtype, self._shift_dtype, self._norm_acc_dtype, self._output_dtype, self._activations, scale, shift)
                elif SupportedNodes.relu6 in self.op_type:
                    params += "static BatchNormReLu6Activation<{:},{:},{:},{:},{:},{:},{:}> {:} = {{\n.m_scale = {{{:}}},\n.m_shift = {{{:}}}}};\n".format(self._pe, self._activation_tiles, self._acc_dtype, self._scale_dtype, self._shift_dtype, self._norm_acc_dtype, self._output_dtype, self._activations, scale, shift)
                else:
                    params += "static BatchNormActivation<{:},{:},{:},{:},{:},{:},{:}> {:} = {{\n.m_scale = {{{:}}},\n.m_shift = {{{:}}}}};\n".format(self._pe, self._activation_tiles, self._acc_dtype, self._scale_dtype, self._shift_dtype, self._norm_acc_dtype, self._output_dtype, self._activations, scale, shift)
            elif SupportedNodes.ins in self.op_type:
                scale_quantizer = WeightQuantizationScheme(q_type='fixed_signed', bit_width=self.scale_bits, int_bit_width=self.scale_int_bits) if self.dtype is Dtypes.FIXED else float
                shift_quantizer = WeightQuantizationScheme(q_type='fixed_signed', bit_width=self.shift_bits, int_bit_width=self.shift_int_bits) if self.dtype is Dtypes.FIXED else float
                scale = torch.from_numpy(self.scale)
                scale = scale_quantizer(scale) if self.dtype is Dtypes.FIXED else scale
                shift = torch.from_numpy(self.shift)
                shift = shift_quantizer(shift) if self.dtype is Dtypes.FIXED else shift
                scale = conv_bias_mapper_2D(scale, self.pe, self.scale_bits, self.scale_int_bits)
                shift = conv_bias_mapper_2D(shift, self.pe, self.shift_bits, self.shift_int_bits)
                if SupportedNodes.relu in self.op_type:
                    params += "static InstanceNormReLuActivation<{:},{:},{:},{:},{:},{:},{:}> {:} = {{\n.m_weight = {{{:}}},\n.m_bias = {{{:}}}}};\n".format(self._pe, self._activation_tiles, self._acc_dtype, self._scale_dtype, self._shift_dtype, self._norm_acc_dtype, self._output_dtype, self._activations, scale, shift)
                elif SupportedNodes.relu6 in self.op_type:
                    params += "static InstanceNormReLu6Activation<{:},{:},{:},{:},{:},{:},{:}> {:} = {{\n.m_weight = {{{:}}},\n.m_bias = {{{:}}}}};\n".format(self._pe, self._activation_tiles, self._acc_dtype, self._scale_dtype, self._shift_dtype, self._norm_acc_dtype, self._output_dtype, self._activations, scale, shift)
                else:
                    params += "static InstanceNormActivation<{:},{:},{:},{:},{:},{:},{:}> {:} = {{\n.m_weight = {{{:}}},\n.m_bias = {{{:}}}}};\n".format(self._pe, self._activation_tiles, self._acc_dtype, self._scale_dtype, self._shift_dtype, self._norm_acc_dtype, self._output_dtype, self._activations, scale, shift)
            elif SupportedNodes.relu in self.op_type:
                params += f"static ReLuActivation<{self._pe}, {self._activation_tiles}, {self._acc_dtype}, {self._output_dtype}"
                if self.dtype is Dtypes.INT:
                    params += f", {self._m0_dtype}> {self._activations} = {{\n.m0 = {self._m0},"
                    if not self.fixed_scale:
                        params += f"\n.sh_val = {self._shift},"
                    params += f"\n.zero_point = {self._output_zero_point}}};\n"
                else:
                    params += f"> {self._activations};\n"
            elif SupportedNodes.relu6 in self.op_type:
                params += f"static ReLu6Activation<{self._pe},{self._activation_tiles}, {self._acc_dtype}, {self._output_dtype}"
                if self.dtype is Dtypes.INT:
                    params += f", {self._m0_dtype}> {self._activations} = {{\n.m0 = {self._m0},"
                    if not self.fixed_scale:
                        params += f"\n.sh_val = {self._shift},"
                    params += f"\n.zero_point = {self._output_zero_point}}};\n"
                else:
                    params += f"> {self._activations};\n"
            else:
                params += f"static PassThroughActivation<{self._acc_dtype}, {self._output_dtype}"
                if self.dtype is Dtypes.INT:
                    params += f", {self._m0_dtype}> {self._activations} = {{\n.m0 = {self._m0},"
                    if not self.fixed_scale:
                        params += f"\n.sh_val = {self._shift},"
                    params += f"\n.zero_point = {self._output_zero_point},\n.min_val = ACT_MIN_VALUE,\n.max_val = ACT_MAX_VALUE}};\n"
                else:
                    params += f"> {self._activations};\n"
        return params

    def macros_def(self, *args):
        string = self.name.upper()
        for arg in args:
            string += arg.upper()
        return string

    def dtype_def(self, *args):
        string = self.name.lower()
        for arg in args:
            string += arg.lower()
        return string

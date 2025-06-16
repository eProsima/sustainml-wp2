""" This file contains the functions responsible for generating the definition of the layers in the model."""


def gen_top_level_def(first_layer_name):
    return ('''void top_level(stream_type<INPUT_MEM_WIDTH> *input,stream_type<OUTPUT_MEM_WIDTH> *output){
        #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
        #pragma HLS INTERFACE mode=m_axi offset=slave port=input  bundle=input depth=num_packed_inputs
        #pragma HLS INTERFACE mode=s_axilite port=input bundle=control
        #pragma HLS INTERFACE mode=m_axi offset=slave port=output  bundle=output depth=num_packed_outputs
        #pragma HLS INTERFACE mode=s_axilite port=output bundle=control
        #pragma HLS DATAFLOW

    hls::stream<stream_type<INPUT_MEM_WIDTH>> m2s_out("m2s_out");
#pragma HLS STREAM variable=m2s_out depth=2
    Mem2Stream<stream_type<INPUT_MEM_WIDTH>,num_packed_inputs>(input,m2s_out);\n'''
f'  hls::stream<stream_type<{first_layer_name}_IA_BITS*{first_layer_name}_SIMD>> sdwc_output_stream("sdwc_output_stream");\n'
'#pragma HLS STREAM variable=sdwc_output_stream depth=2\n'
f'  StreamingDataWidthConverter_Batch<INPUT_MEM_WIDTH,{first_layer_name}_IA_BITS*{first_layer_name}_SIMD,num_packed_inputs>(m2s_out,sdwc_output_stream,1);\n\n')

def gen_bottom_level_def(last_layer_name):
    return (f'    hls::stream<stream_type<OUTPUT_MEM_WIDTH>> out_sdwc_output_stream("out_sdwc_output_stream");\n'
            f'#pragma HLS STREAM variable = out_sdwc_output_stream depth = 2\n'
            f'    StreamingDataWidthConverter_Batch<{last_layer_name.upper()}_OA_BITS * {last_layer_name.upper()}_PE, OUTPUT_MEM_WIDTH, {last_layer_name.upper()}_OUT_FEATURES>({last_layer_name.lower()}_output_stream, out_sdwc_output_stream, reps);\n'
            f'    Stream2Mem<stream_type<OUTPUT_MEM_WIDTH>, num_packed_outputs>(out_sdwc_output_stream, output);\n'
            f'}}\n#endif\n')

def gen_top_level_hpp_def():
    return (f'#ifndef TOP_HPP\n'
            f'#define TOP_HPP\n'
            f'#include "config.hpp"\n'
            f'void top_level(stream_type<INPUT_MEM_WIDTH> *input,stream_type<OUTPUT_MEM_WIDTH> *output);\n'
            f'#endif\n')


def gen_finn_conv2d_def(counter, prev_layer_name):
    assert prev_layer_name is not None
    return (f'    hls::stream<stream_type<CONV_{counter}_PE* conv_{counter}_output_dtype::width>> conv_{counter}_output_stream("conv_{counter}_output_stream");\n'
            f'#pragma HLS STREAM variable = conv_{counter}_output_stream depth = 10\n'
            f'    ConvLayer_Batch<CONV_{counter}_K, CONV_{counter}_IFM_CH, CONV_{counter}_IFM_DIM + CONV_{counter}_PADDING * 2, CONV_{counter}_OFM_CH, CONV_{counter}_OFM_DIM, CONV_{counter}_SIMD, CONV_{counter}_PE,\n'
            f'        Slice<conv_{counter}_input_dtype>, Slice<conv_{counter}_output_dtype>, Identity>(\n'
            f'            {prev_layer_name}_output_stream,\n'
            f'            conv_{counter}_output_stream,\n'
            f'            conv_{counter}_weights,\n'
            f'            conv_{counter}_biases,\n'
            f'            reps,\n'
            f'            ap_resource_dflt());\n\n')


def gen_streaming_max_pool_def(counter, prev_layer_name, prev_type_modifier):
    assert prev_layer_name is not None
    return (f'    hls::stream<stream_type<MAXPOOL_{counter}_PE * {prev_type_modifier}_output_dtype::width>> maxpool_{counter}_output_stream("maxpool_{counter}_output_stream");\n'
            f'#pragma HLS STREAM variable = maxpool_{counter}_output_stream depth = 10\n'
            f'    Streaming_Maxpool_batch2d<MAXPOOL_{counter}_IFM_CH, MAXPOOL_{counter}_IFM_DIM, MAXPOOL_{counter}_PE, MAXPOOL_{counter}_K, MAXPOOL_{counter}_STRIDE,\n'
            f'        Slice<{prev_type_modifier}_output_dtype>, {prev_type_modifier}_output_dtype, {prev_layer_name.upper()}_PE * {prev_type_modifier}_output_dtype::width, MAXPOOL_{counter}_PE * {prev_type_modifier}_output_dtype::width>\n'
            f'        ({prev_layer_name}_output_stream, maxpool_{counter}_output_stream, reps);\n\n')


def gen_fmpadding_def(padded_layer_name, prev_layer_name):
    return (f'    hls::stream<stream_type<{padded_layer_name}_SIMD * {padded_layer_name.lower()}_input_dtype::width>> {padded_layer_name.lower()}_padding_output_stream("{padded_layer_name.lower()}_padding_output_stream");\n'
            f'#pragma HLS STREAM variable = {padded_layer_name.lower()}padding_output_stream depth = 4\n'
            f'    FMPadding_Batch<{padded_layer_name}IFM_DIM,\n'
            f'        {padded_layer_name}_FM_DIM + {padded_layer_name}_PADDING * 2,\n'
            f'        {padded_layer_name}_ADDING * 2,\n'
            f'        {padded_layer_name}_FM_CH,\n'
            f'        {padded_layer_name}_IMD,\n'
            f'        {padded_layer_name.lower()}_input_dtype>\n'
            f'        ({prev_layer_name}_output_stream, {padded_layer_name.lower()}_padding_output_stream, reps);\n\n')


def gen_linear_def(counter, prev_layer_name):
    assert prev_layer_name is not None
    return (f'    hls::stream<stream_type<FC_{counter}_PE* fc_{counter}_output_dtype::width>> fc_{counter}_output_stream("fc_{counter}_output_stream");\n'
            f'#pragma HLS STREAM variable = fc_{counter}_output_stream depth = 10\n'
            f'    StreamingFCLayer_Batch<FC_{counter}_IN_FEATURES, FC_{counter}_OUT_FEATURES, FC_{counter}_SIMD, FC_{counter}_PE,\n'
            f'        Slice<fc_{counter}_input_dtype>, Slice<fc_{counter}_output_dtype>, Identity>(\n'
            f'            {prev_layer_name}_output_stream,\n'
            f'            fc_{counter}_output_stream,\n'
            f'            fc_{counter}_weights,\n'
            f'            fc_{counter}_biases,\n'
            f'            reps,\n'
            f'            ap_resource_dflt());\n\n')


def gen_duplicate_stream_def(source_name,dst_1_name,dst_2_name):
    return(f'    hls::stream<stream_type<{source_name.upper()}_PE* {source_name}_output_dtype::width>> {dst_1_name}_input_stream("{dst_1_name}_input_stream");\n'
           f'#pragma HLS STREAM variable = {dst_1_name}input_stream depth = 10\n'
           f'    hls::stream<stream_type<{source_name.upper()}_PE* {source_name}_output_dtype::width>> {dst_2_name}_input_stream("{dst_2_name}_input_stream");\n'
           f'#pragma HLS STREAM variable = {dst_2_name}_input_stream depth = 10\n'
           f'   DuplicateStreams<{source_name.upper()}_PE* {source_name}_output_dtype::width, {source_name.upper()}_OFM_DIM * {source_name.upper()}_OFM_DIM * ({source_name.upper()}_OFM_CH / {source_name.upper()}_PE)>(\n'
           f'   {source_name}_output_stream,\n'
           f'   {dst_1_name}_input_stream,\n'
           f'   {dst_2_name}_input_stream);)\n\n')


def gen_norse_lif_def(counter, prev_layer_name):
    assert prev_layer_name is not None
    return (f'    hls::stream<stream_type<{prev_layer_name.lower()}_output_dtype::width>> lif_{counter}_output_stream("lif_{counter}_output_stream");\n'
            f'#pragma HLS STREAM variable = lif_{counter}_output_stream depth = 10\n'
            f'    Lif_Snn<LIF_{counter}_NUM_NEURONS,\n'
            f'        SEQ_LENGTH,\n'
            f'        LIF_{counter}_TAU_MEM_INV,\n'
            f'        LIF_{counter}_TAU_SYN_INV,\n'
            f'        {prev_layer_name.lower()}_output_dtype>({prev_layer_name}_output_stream, lif_{counter}_output_stream, LIF_{counter}_DT, LIF_{counter}_V_TH);\n\n')


def gen_norse_cclif_encoder_def(counter, prev_layer_name, prev_type_modifier):
    assert prev_layer_name is not None and prev_type_modifier is not None
    return (f'    hls::stream<stream_type<{prev_type_modifier}_output_dtype::width>> cclif_{counter}_output_stream("cclif_{counter}_output_stream");\n'
            f'#pragma HLS STREAM variable = cclif_{counter}_output_stream depth = 10\n'
            f'    Constant_Current_Lif_Encoder<CCLIF_{counter}_NUM_NEURONS,\n'
            f'        CCLIF_{counter}_SEQ_LENGTH,\n'
            f'        CCLIF_{counter}_TAU_MEM_INV,\n'
            f'        CCLIF_{counter}_TAU_SYN_INV,\n'
            f'        {prev_type_modifier}_input_dtype>({prev_layer_name}_output_stream, cclif_{counter}_output_stream, CCLIF_{counter}_DT, CCLIF_{counter}_V_TH);\n\n')


def gen_norse_lilinear_cell_def(counter, prev_layer_name):
    assert prev_layer_name is not None
    return (f'    hls::stream<stream_type<LIL_{counter}_PE * lil_{counter}_output_dtype::width>> lil_{counter}_output_stream("lil_{counter}_output_stream");\n'
            f'#pragma HLS STREAM variable = lil_{counter}_output_stream depth = 10\n'
            f'    LILinearCell<LIL_{counter}_IN_FEATURES,\n'
            f'        LIL_{counter}_OUT_FEATURES,\n'
            f'        SEQ_LENGTH, LIL_{counter}_SIMD,\n'
            f'        LIL_{counter}_PE,\n'
            f'        Slice<lil_{counter}_input_dtype>, Slice<lil_{counter}_output_dtype>, Identity>\n'
            f'        ({prev_layer_name}_output_stream, lil_{counter}_output_stream, LIL_{counter}_TAU_MEM_INV, LIL_{counter}_TAU_SYN_INV, lil_{counter}_weights,ap_resource_dflt());\n\n')


def gen_add_streams_def(counter, src_1_name, src_2_name, prev_type_modifier, total_words):
    assert src_1_name is not None and src_2_name is not None and prev_type_modifier is not None
    return (f'    hls::stream<stream_type<{prev_type_modifier}output_dtype::width>> add_{counter}_output_stream("add_{counter}_output_stream");\n'
            f'#pragma HLS STREAM variable = add_{counter}_output_stream depth = 10\n'
            f'    AddStreams<{src_1_name}_PE,\n'
            f'        {prev_type_modifier}_output_dtype,\n'
            f'        {prev_type_modifier}_output_dtype,\n'
            f'        {prev_type_modifier}_output_dtype,\n'
            f'        {total_words}>({src_1_name.lower()}_output_stream, {src_2_name.lower()}_output_stream, add_{counter}_output_stream);\n\n')


def gen_norse_lif_recurrent_cell_def(counter, prev_layer_name):
    assert prev_layer_name is not None
    return (f'    hls::stream<stream_type<LIFR_{counter}_PE * lifr_{counter}_output_dtype::width>> lifr_{counter}_output_stream("lifr_{counter}_output_stream");\n'
            f'#pragma HLS STREAM variable = lifr_{counter}_output_stream depth = 10\n'
            f'    LIFRecurrentCell<LIFR_{counter}_IN_FEATURES,\n'
            f'        LIFR_{counter}_OUT_FEATURES,\n'
            f'        SEQ_LENGTH, lifr_{counter}_output_dtype,\n'
            f'        LIFR_{counter}_SIMD,\n'
            f'        LIFR_{counter}_PE,\n'
            f'        Slice<lifr_{counter}_input_dtype>, Slice<lifr_{counter}_output_dtype>, Identity>\n'
            f'        ({prev_layer_name}_output_stream, lifr_{counter}_output_stream, LIFR_{counter}_TAU_MEM_INV, LIFR_{counter}_TAU_SYN_INV,\n'
            f'        LIFR_{counter}_V_TH, LIFR_{counter}_V_RESET, LIFR_{counter}_V_LEAK,\n'
            f'        lifr_{counter}_input_weights, lifr_{counter}_recurrent_weights, ap_resource_dflt());\n\n')
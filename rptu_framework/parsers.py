''' 
    This file contains functions responsible for parsing the layers of the model and generating the corresponding C++ parameters
'''


def conv2d_parser(op_type, node_attr, input_shape, output_shape, config_file, count, pe, simd, relu=False):
    ''' Generates the configuration parameters for the Conv2d layer
    Args:
        op_type: Operation type of the layer {Conv, TransposedConv}
        node_attr: dictionary containing the attributes of the node, it contains {kernel_shape,strides,pads,group}
        file: the file object to write the parameters to
        count: the index of the node in the model graph
        pe: number of pes to use for this layer
        simd: number of simds to use for this layer
        fixed_point: weather to use fixed point or integer, if integer is used scale is printed instead of the actual number of int bits
        conv: boolean to indicate if the layer is a QuantConv2d or TransposedConv2d layer, True for QuantConv2d and False for TransposedConv2d
    '''
    assert input_shape[1] % simd == 0, f'input channels {input_shape[1]} is not divisible by simd {simd}'
    assert output_shape[1] % pe == 0, f'output channels {output_shape[1]} is not divisible by pe {pe}'
    layer_name = f'{op_type}_{count}_'.upper()
    config_file.write(f'{f"#define {layer_name}K":<48}{node_attr["kernel_shape"][0]}\n')
    config_file.write(f'{f"#define {layer_name}STRIDE":<48}{node_attr["strides"][0]}\n')
    config_file.write(f'{f"#define {layer_name}PADDING":<48}{node_attr["pads"][0]*2}\n')
    config_file.write(f'{f"#define {layer_name}GROUPS":<48}{node_attr["group"]}\n')
    config_file.write(f'{f"#define {layer_name}IFM_CH":<48}{input_shape[1]}\n')
    config_file.write(f'{f"#define {layer_name}IFM_DIM":<48}{input_shape[2]}\n')
    config_file.write(f'{f"#define {layer_name}OFM_CH":<48}{output_shape[1]}\n')
    config_file.write(f'{f"#define {layer_name}OFM_DIM":<48}{output_shape[2]}\n')
    config_file.write(f'{f"#define {layer_name}SIMD":<48}{simd}\n')
    config_file.write(f'{f"#define {layer_name}PE":<48}{pe}\n')
    config_file.write(f'{f"#define {layer_name}RELU":<48}{int(relu)}\n')
    config_file.write(f'{f"#define {layer_name}WEIGHT_BITS":<48}{16}\n')
    config_file.write(f'{f"#define {layer_name}WEIGHT_INT_BITS":<48}{8}\n')
    config_file.write(f'{f"#define {layer_name}BIAS_BITS":<48}{16}\n')
    config_file.write(f'{f"#define {layer_name}BIAS_INT_BITS":<48}{8}\n')
    config_file.write(f'{f"#define {layer_name}IA_BITS":<48}{16}\n')
    config_file.write(f'{f"#define {layer_name}IA_INT_BITS":<48}{8}\n')
    config_file.write(f'{f"#define {layer_name}OA_BITS":<48}{16}\n')
    config_file.write(f'{f"#define {layer_name}OA_INT_BITS":<48}{8}\n')
    config_file.write(f'{f"#define {layer_name}BMEM":<48}{ output_shape[1] // pe}\n')
    config_file.write(f'{f"#define {layer_name}WMEM":<48}{ (node_attr["kernel_shape"][0] * node_attr["kernel_shape"][0] * input_shape[1] * output_shape[1]) // (pe * simd)}\n')
    config_file.write(f'typedef ap_fixed<{layer_name}WEIGHT_BITS,{layer_name}WEIGHT_INT_BITS,AP_RND_ZERO,AP_WRAP> {layer_name.lower()}weight_dtype;\n')
    config_file.write(f'typedef ap_fixed<{layer_name}BIAS_BITS,{layer_name}BIAS_INT_BITS,AP_RND_ZERO,AP_WRAP> {layer_name.lower()}bias_dtype;\n')
    config_file.write(f'typedef ap_fixed<{layer_name}IA_BITS,{layer_name}IA_INT_BITS,AP_RND_ZERO,AP_WRAP> {layer_name.lower()}input_dtype;\n')
    config_file.write(f'typedef ap_fixed<{layer_name}OA_BITS,{layer_name}OA_INT_BITS,AP_RND_ZERO,AP_WRAP> {layer_name.lower()}output_dtype;\n')
    config_file.write('\n\n')


def pool2d_parser(op_type, node_attr, input_shape, output_shape, config_file, counter, pe, simd):
    layer_name = f'{op_type}_{counter}_'.upper()
    config_file.write(f'{f"#define {layer_name}K":<48}{node_attr["kernel_shape"][0]}\n')
    config_file.write(f'{f"#define {layer_name}STRIDE":<48}{node_attr["strides"][0]}\n')
    if 'pads' in node_attr:
        config_file.write(f'{f"#define {layer_name}PADDING":<48}{node_attr["pads"][0]*2}\n')
    config_file.write(f'{f"#define {layer_name}IFM_CH":<48}{input_shape[1]}\n')
    config_file.write(f'{f"#define {layer_name}IFM_DIM":<48}{input_shape[2]}\n')
    config_file.write(f'{f"#define {layer_name}OFM_CH":<48}{output_shape[1]}\n')
    config_file.write(f'{f"#define {layer_name}OFM_DIM":<48}{output_shape[2]}\n')
    config_file.write(f'{f"#define {layer_name}SIMD":<48}{simd}\n')
    config_file.write(f'{f"#define {layer_name}PE":<48}{pe}\n')
    config_file.write('\n\n')


def relu_parser(input_shape, output_shape, config_file, counter, pe):
    config_file.write(f'{f"#define RELU_{counter}_IFM_CH":<48}{input_shape[1]}\n')
    config_file.write(f'{f"#define RELU_{counter}_IFM_DIM":<48}{input_shape[2]}\n')
    config_file.write(f'{f"#define RELU_{counter}_OFM_CH":<48}{output_shape[1]}\n')
    config_file.write(f'{f"#define RELU_{counter}_OFM_DIM":<48}{output_shape[2]}\n')
    config_file.write(f'{f"#define RELU_{counter}_PE":<48}{pe}\n')
    config_file.write('\n\n')


def fullyconn_parser(input_shape, output_shape, config_file, counter, pe, simd, relu=False):
    assert input_shape[1] % simd == 0, f'input channels {input_shape[1]} is not divisible by simd {simd}'
    assert output_shape[1] % pe == 0, f'output channels {output_shape[1]} is not divisible by pe {pe}'

    config_file.write(f'{f"#define FC_{counter}_IN_FEATURES":<48}{input_shape[1]}\n')
    config_file.write(f'{f"#define FC_{counter}_OUT_FEATURES":<48}{output_shape[1]}\n')
    config_file.write(f'{f"#define FC_{counter}_SIMD":<48}{simd}\n')
    config_file.write(f'{f"#define FC_{counter}_PE":<48}{pe}\n')
    config_file.write(f'{f"#define FC_{counter}_RELU":<48}{int(relu)}\n')
    config_file.write(f'{f"#define FC_{counter}_BMEM":<48}{ output_shape[1] // pe}\n')
    config_file.write(f'{f"#define FC_{counter}_WMEM":<48}{ (input_shape[1] * output_shape[1]) // (pe * simd)}\n')
    config_file.write(f'{f"#define FC_{counter}_WEIGHT_BITS":<48}{ 16}\n')
    config_file.write(f'{f"#define FC_{counter}_WEIGHT_INT_BITS":<48}{ 8}\n')
    config_file.write(f'{f"#define FC_{counter}_BIAS_BITS":<48}{ 16}\n')
    config_file.write(f'{f"#define FC_{counter}_BIAS_INT_BITS":<48}{ 8}\n')
    config_file.write(f'{f"#define FC_{counter}_IA_BITS":<48}{ 16}\n')
    config_file.write(f'{f"#define FC_{counter}_IA_INT_BITS":<48}{ 8}\n')
    config_file.write(f'{f"#define FC_{counter}_OA_BITS":<48}{ 16}\n')
    config_file.write(f'{f"#define FC_{counter}_OA_INT_BITS":<48}{ 8}\n')
    config_file.write(f'typedef ap_fixed<FC_{counter}_WEIGHT_BITS,FC_{counter}_WEIGHT_INT_BITS,AP_RND_ZERO,AP_WRAP> fc_{counter}_weight_dtype;\n')
    config_file.write(f'typedef ap_fixed<FC_{counter}_BIAS_BITS,FC_{counter}_BIAS_INT_BITS,AP_RND_ZERO,AP_WRAP> fc_{counter}_bias_dtype;\n')
    config_file.write(f'typedef ap_fixed<FC_{counter}_IA_BITS,FC_{counter}_IA_INT_BITS,AP_RND_ZERO,AP_WRAP> fc_{counter}_input_dtype;\n')
    config_file.write(f'typedef ap_fixed<FC_{counter}_OA_BITS,FC_{counter}_OA_INT_BITS,AP_RND_ZERO,AP_WRAP> fc_{counter}_output_dtype;\n')
    config_file.write('\n\n')


def lil_lifr_cell_parser(layer, config_file, name, pe, simd):
    ''' Generates the config for LIFRecurrentCell and LILinearCell
    Args:
        layer: the layer to be parsed
        config_file: the config file to be written
        name: the name of the layer
        pe: the number of processing elements
        simd: the number of SIMD lanes
    '''
    assert layer.input_size % simd == 0, f'input channels {layer.input_size} is not divisible by simd {simd}'
    assert layer.hidden_size % pe == 0, f'output channels {layer.hidden_size} is not divisible by pe {pe}'

    config_file.write(f'{f"#define {name}_IN_FEATURES":<48}{layer.input_size}\n')
    config_file.write(f'{f"#define {name}_OUT_FEATURES":<48}{layer.hidden_size}\n')
    config_file.write(f'{f"#define {name}_SIMD":<48}{simd}\n')
    config_file.write(f'{f"#define {name}_PE":<48}{pe}\n')
    config_file.write(f'{f"#define {name}_TAU_MEM_INV":<48}{layer.p.tau_mem_inv.max().item() * layer.dt}\n')
    config_file.write(f'{f"#define {name}_TAU_SYN_INV":<48}{layer.p.tau_syn_inv.max().item() * layer.dt}\n')
    config_file.write(f'{f"#define {name}_V_TH":<48}{layer.p.v_th.max().item()}\n')
    config_file.write(f'{f"#define {name}_V_LEAK":<48}{layer.p.v_leak.item()}\n')
    config_file.write(f'{f"#define {name}_V_RESET":<48}{layer.p.v_reset.max().item()}\n')
    config_file.write(f'{f"#define {name}_BMEM":<48}{ layer.hidden_size // pe}\n')
    config_file.write(f'{f"#define {name}_WMEM":<48}{ (layer.input_size * layer.hidden_size) // (pe * simd)}\n')
    config_file.write(f'{f"#define {name}_WEIGHT_BITS":<48}{ 16}\n')
    config_file.write(f'{f"#define {name}_WEIGHT_INT_BITS":<48}{ 8}\n')
    config_file.write(f'{f"#define {name}_IA_BITS":<48}{ 16}\n')
    config_file.write(f'{f"#define {name}_IA_INT_BITS":<48}{ 8}\n')
    config_file.write(f'{f"#define {name}_OA_BITS":<48}{ 16}\n')
    config_file.write(f'{f"#define {name}_OA_INT_BITS":<48}{ 8}\n')
    config_file.write(f'typedef ap_fixed<{name}_WEIGHT_BITS,{name}_WEIGHT_INT_BITS,AP_RND_ZERO,AP_WRAP> {name.lower()}_weight_dtype;\n')
    config_file.write(f'typedef ap_fixed<{name}_IA_BITS,{name}_IA_INT_BITS,AP_RND_ZERO,AP_WRAP> {name.lower()}_input_dtype;\n')
    config_file.write(f'typedef ap_fixed<{name}_OA_BITS,{name}_OA_INT_BITS,AP_RND_ZERO,AP_WRAP> {name.lower()}_output_dtype;\n')
    config_file.write('\n\n')


def norse_lif_parser(lif, num_neurons, config_file, counter, relu=False):
    config_file.write(f'{f"#define LIF_{counter}_NUM_NEURONS":<48}{num_neurons}\n')
    config_file.write(f'{f"#define LIF_{counter}_PE":<48}{1}\n')
    config_file.write(f'{f"#define LIF_{counter}_RELU":<48}{int(relu)}\n')
    config_file.write(f'{f"#define LIF_{counter}_TAU_MEM_INV":<48}{int(lif.p.tau_mem_inv.item())}\n')
    config_file.write(f'{f"#define LIF_{counter}_TAU_SYN_INV":<48}{int(lif.p.tau_syn_inv.item())}\n')
    config_file.write(f'{f"#define LIF_{counter}_DT":<48}{lif.dt}\n')
    config_file.write(f'{f"#define LIF_{counter}_V_TH":<48}{lif.p.v_th}\n')
    config_file.write('\n\n')


def norse_cclif_encoder_parser(enc, num_neurons, config_file, counter):
    '''norse ConstantCurrentLIFEncoder parser'''
    config_file.write(f'{f"#define CCLIF_{counter}_NUM_NEURONS":<48}{num_neurons}\n')
    config_file.write(f'{f"#define CCLIF_{counter}_PE":<48}{1}\n')
    config_file.write(f'{f"#define CCLIF_{counter}_SEQ_LENGTH":<48}{enc.seq_length}\n')
    config_file.write(f'{f"#define CCLIF_{counter}_TAU_MEM_INV":<48}{int(enc.p.tau_mem_inv.item())}\n')
    config_file.write(f'{f"#define CCLIF_{counter}_TAU_SYN_INV":<48}{int(enc.p.tau_syn_inv.item())}\n')
    config_file.write(f'{f"#define CCLIF_{counter}_DT":<48}{enc.dt}\n')
    config_file.write(f'{f"#define CCLIF_{counter}_V_TH":<48}{enc.p.v_th}\n')
    config_file.write('\n\n')

"""
This files contains functions used to integratethe generator into the SustainML framework.
"""
from .network_hls import NetworkHLS
from .defaults import Dtypes, DefaultValues
import networkx as nx

DEFAULT_INP_MEM_WIDTH = 128
DEFAULT_OUT_MEM_WIDTH = 128
DEFAULT_INTER_STREAMS_DEPTH = 128
def onnx_ml_resource_estimation(onnx_path, device, target_fps=25):
    networkhls = NetworkHLS(dtype=Dtypes.INT, input_mem_width=DEFAULT_INP_MEM_WIDTH, output_mem_width=DEFAULT_OUT_MEM_WIDTH, target_fps=target_fps, intermediate_streams_depth=DEFAULT_INTER_STREAMS_DEPTH)
    nx_graph = networkhls.onnx_to_networkx(model_path=onnx_path, brevitas=False, int_quant=True, scale_bit_width=32, stop_node=None, fixed_scale=False)
    res_dict = dict()
    total_latency_cc = 0
    for node in nx_graph.nodes():
        total_latency_cc += max(node.get_latency_cc_node()) + node.get_depth_cc_node()
    latency = total_latency_cc * (1 / DefaultValues.freq_mhz)
    res_dict['Latency'] = latency
    res_dict['Total_LUTs'] = 0
    res_dict['Logic_LUTs'] = 0
    res_dict['LUTRAMs'] = 0
    res_dict['SRLs'] = 0
    res_dict['FFs'] = 0
    res_dict['RAMB36'] = 0
    res_dict['RAMB18'] = 0
    res_dict['URAM'] = 0
    res_dict['DSP_Blocks'] = 0
    res_dict['Ideal_power'] = 0
    res_dict['Run_power'] = 0
    return res_dict

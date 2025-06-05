"""
Module: onnx_to_hls

This module provides a command-line interface for converting ONNX models to High-Level Synthesis (HLS) code.
It performs various transformations and optimizations on the ONNX model,
such as removing unsupported nodes, adding split nodes, merging nodes, setting FIFO depth, and dividing the model into multiple top_levels.
The final HLS model is saved as .hpp and .cpp files.

Classes:
    NetworkHLS: A class that provides methods to convert ONNX models to HLS models.

Functions:
    main(): Parses command-line arguments and performs the conversion from ONNX to HLS.

Command-line Arguments:
    --model: Path to an ONNX model (required).
    --result_hpp_path: Path to save the generated .hpp files (default: './include_hw/').
    --result_cpp_path: Path to save the generated .cpp files (default: './src_hw/').
    --input_mem_width: Input memory width (default: 64).
    --output_mem_width: Output memory width (default: 64).
    --max_subgraph_weight: Maximum subgraph weight (default: 25e4).
    --subgraph_weight_margin: Allowed subgraph weight margin (default: 1.1).
    --brevitas: Flag indicating if the model is a Brevitas model (default: False).
    --int: Flag indicating if the model is a quantized ONNX model using QDQ method (default: False).
    --float: Flag indicating if the model should be treated as a float model (default: False).
    --scale_bit_width: Bit width for scale (default: 16).
    --profiler_json: Path to the profiler JSON file (default: './profiled_activations.json').
    --stop_node: Node that will be the last node in the graph (default: '').
"""
import argparse
from pathlib import Path
from network_hls import NetworkHLS
from .defaults import Dtypes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic ONNX based HLS model generator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='path to an ONNX model')
    parser.add_argument('--result_hpp_path', type=Path, default='./include_hw/', help='path to save the generated .hpp files')
    parser.add_argument('--result_cpp_path', type=Path, default='./src_hw/', help='path to save the generated .cpp files')
    parser.add_argument('--input_mem_width', type=int, help='input memory width ', default=128)
    parser.add_argument('--output_mem_width', type=int, help='output memory width', default=128)
    parser.add_argument('--max_subgraph_weight', type=float, help='max subgraph weight', default=25e4)
    parser.add_argument('--subgraph_weight_margin', type=float, help='allowed subgraph weight margin', default=1.1)
    parser.add_argument('--brevitas', action='store_true', help='brevitas model', default=False)
    parser.add_argument('--int', action='store_true', help='Quant onnx model quantized with QDQ method', default=False)
    parser.add_argument('--float', action='store_true', help='treat the model as float', default=False)
    parser.add_argument('--scale_bit_width', type=int, help='bit width for scale', default=16)
    parser.add_argument('--fixed_scale', action='store_true', help='use fixed scale', default=False)
    parser.add_argument('--profiler_json', type=str, help='path to the profiler json file', default='./profiled_activations.json')
    parser.add_argument('--stop_node', type=str, help='This node will be the last node in the graph', default='')
    parser.add_argument('--quiet', action='store_true', help='not prining, no logging, no progress par, no image generation', default=False)  # TODO: use it to stop logging and progress bar
    parser.add_argument('--max_on_chip_depth', type=int, help='max on chip depth for skip connection', default=2e9)
    parser.add_argument('--set_fifos_to_max', action='store_true', help='set the size of skip connections to the maximam value', default=False)
    parser.add_argument('--fifos_size_percentage', type=float, help='sets the size of skip connections to a percentage of the maximam value', default=None)
    parser.add_argument('--target_fps', type=int, help='target fps', default=25)
    parser.add_argument('--intermediate_streams_depth', type=int, help='intermediate streams depth', default=2)

    args = parser.parse_args()
    args.model = args.model.resolve()
    args.result_hpp_path = args.result_hpp_path.resolve()
    args.result_cpp_path = args.result_cpp_path.resolve()
    assert args.model.is_file(), 'Model file does not exist'
    assert not(args.int and args.float), 'Int and float flags cannot be used together'
    if args.set_fifos_to_max:
        assert args.fifos_size_percentage is None, 'fifos_size_percentage and set_fifos_to_max flags cannot be used together'
    if args.int:
        dtype = Dtypes.INT
    elif args.float:
        dtype = Dtypes.FLOAT
    else:
        dtype = Dtypes.FIXED
    networkhls = NetworkHLS(dtype=dtype, input_mem_width=args.input_mem_width, output_mem_width=args.output_mem_width, target_fps=args.target_fps, intermediate_streams_depth=args.intermediate_streams_depth)
    nx_graph = networkhls.onnx_to_networkx(args.model, args.brevitas, args.int, args.scale_bit_width, args.stop_node, args.fixed_scale)

    # Visualize the graph
    if not args.quiet:
        networkhls.networkx_to_png(nx_graph, "after_onnx")

    # Remove unsupported layers, currently "Reshape", "Transpose", ..
    networkhls.remove_unsupported_nodes(nx_graph)
    if not args.quiet:
        networkhls.networkx_to_png(nx_graph, "after_removing_unsupported_nodes")

    # Set a parallelism of the layers
    nx_graph = networkhls.set_parallelism(nx_graph, chain_rule=False)
    if not args.quiet:
        networkhls.networkx_to_png(nx_graph, "after_setting_parallelism")
    # Add Split layer
    nx_graph = networkhls.add_split_nodes(nx_graph)
    #networkhls.networkx_to_png(nx_graph, "after_adding_split")

    # Merge several layers to one, like
    # [Conv/Linear Bn/In, Relu]
    # [Conv/Linear, Bn/In]
    # [Conv/Linear, Relu]
    nx_graph = networkhls.merge_several_nodes(nx_graph)
    #networkhls.networkx_to_png(nx_graph, "after_merging_nodes")

    # Set a depth of the FIFOs on a path between Split and Merge
    nx_graph = networkhls.set_fifo_depth(nx_graph, max_depth=args.max_on_chip_depth, set_to_max=args.set_fifos_to_max, fifos_size_percentage=args.fifos_size_percentage)
    if not args.quiet:
        networkhls.networkx_to_png(nx_graph, "after_setting_fifo_depth")

    nx_graph = networkhls.add_sdwc_nodes(nx_graph)
    if not args.quiet:
        networkhls.networkx_to_png(nx_graph, "after_adding_sdwc_nodes")
    # Add sdwc nodes if needed

    networkhls.update_nodes_default(nx_graph, args.brevitas)
    if not args.quiet:
        networkhls.networkx_to_png(nx_graph, "after_updating_nodes")
    # Create subgraphs
    networkhls.create_single_subgraph_default(nx_graph)
    #networkhls.create_subgraphs_default(nx_graph)
    #networkhls.create_subgraphs(nx_graph, args.max_subgraph_weight, args.subgraph_weight_margin)
    if not args.quiet:
        networkhls.networkx_to_png(nx_graph, "after_creating_subgraphs")

    networkhls.generate_top_level_wrapper_hpp(nx_graph, args.result_hpp_path / 'dnn_top_level_wrapper.hpp')

    networkhls.generate_top_level_wrapper_cpp(nx_graph, args.result_cpp_path / 'dnn_top_level_wrapper.cpp')

    networkhls.generate_top_level_cpp(nx_graph, args.result_cpp_path / 'dnn_top_level.cpp')

    # Generate profilers file
    networkhls.generate_profilers_hpp_cpp(nx_graph, args.result_hpp_path / 'dnn_profilers.hpp', args.result_cpp_path / 'dnn_profilers.cpp')

    # Generate configuration file
    networkhls.generate_config_hpp(nx_graph, args.result_hpp_path / 'dnn_config.hpp', args.scale_bit_width)

    # Generate params file
    networkhls.generate_params_hpp(args.result_hpp_path / 'dnn_params.hpp')

    if args.int:
        exit(0)
    print('\nDo you have a valid .json file with profiled activations? Would you like to continue....press any key or abort Ctrl+C')
    input()

    # Update the attributes with values based on hardware profiling
    nx_graph = networkhls.json_params_to_networkx(nx_graph, './profiled_activations.json')

    networkhls.update_nodes(nx_graph)
    networkhls.generate_top_level_wrapper_hpp(nx_graph, args.result_hpp_path /'dnn_top_level_wrapper.hpp')

    networkhls.generate_top_level_wrapper_cpp(nx_graph, args.result_cpp_path /'dnn_top_level_wrapper.cpp')

    networkhls.generate_top_level_cpp(nx_graph, args.result_cpp_path /'dnn_top_level.cpp')

    # Generate profilers file
    networkhls.generate_profilers_hpp_cpp(nx_graph, args.result_hpp_path / 'dnn_profilers.hpp', args.result_cpp_path / 'dnn_profilers.cpp')

    # Generate configuration file
    networkhls.generate_config_hpp(nx_graph, args.result_hpp_path / 'dnn_config.hpp')

    # Generate params file
    networkhls.generate_params_hpp(args.result_hpp_path / 'dnn_params.hpp')

import torch
import yaml
import os
import onnx
import onnxsim
import onnxruntime
from onnxruntime import quantization
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

def torch_to_onnx(torch_model, output_file, input_channels, input_size):
    torch_model.eval()
    input_shape = (1, input_channels, input_size, input_size)
    input_name = 'input'
    example_input = torch.randn(input_shape)
    #torch.onnx.enable_fake_mode()
    # export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    # onnx_program = torch.onnx.dynamo_export(torch_model, example_input, export_options=export_options)
    torch.onnx.export(
             torch_model.cpu(),
             example_input.cpu(),
             output_file,
             do_constant_folding=True,
             keep_initializers_as_inputs=True,
             opset_version=20,
             input_names=[input_name],
             output_names=['output']
         )
    model_onnx = onnx.load(output_file)
    onnx.checker.check_model(model_onnx)
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, 'assert check failed'
    onnx.save(model_onnx, output_file)
    quantize_onnx(output_file, input_name, input_shape)

def run_onnx(onnx_path, input_data):
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_data)}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs[0]

def compare_outputs(torch_model, onnx_path, initial_channels, input_size):
    torch_model.eval()
    input_data = torch.randn(1, initial_channels, input_size, input_size)
    torch_output = torch_model(input_data)
    onnx_output = run_onnx(onnx_path, input_data)
    torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=1e-3, atol=1e-3)

class DataReader(CalibrationDataReader):
    ''' Dummy data reader for quantization.
    '''
    def __init__(self, num_samples, input_name, input_shape):
        # Use inference session to get input shape.
        self.input_name = input_name
        self.datasize = num_samples
        self.counter = 0
        self.num_samples = num_samples
        self.input_shape = input_shape

    def get_next(self):
        img = torch.rand(self.input_shape)
        img = img.numpy()
        sample = {self.input_name: img}
        self.counter += 1
        if self.counter == self.num_samples:
            return None
        return sample

    def rewind(self):
        self.counter = 0

def quantize_onnx(onnx_path, input_name, input_shape, num_samples=50, output_post_fix=''):
    pre_processed_model = f'{onnx_path[:-5]}-preprocessed.onnx'
    quantized_model = f'{onnx_path[:-5]}{output_post_fix}.onnx'
    # onnx_model = onnx.load(onnx_path)
    quantization.shape_inference.quant_pre_process(onnx_path, pre_processed_model, skip_symbolic_shape=False)
    print("Model preprocessed successfully!")
    data_reader = DataReader(num_samples, input_name, input_shape)
    print("Data reader created successfully!")
    print("Quantizing model...")
    quantize_static(
    pre_processed_model,
    quantized_model,
    data_reader,
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    activation_type = QuantType.QInt8,
    weight_type=QuantType.QInt8,
    reduce_range=True)
    onnx.checker.check_model(quantized_model, full_check=True)
    print("Model quantized successfully and saved as: ", quantized_model)
    os.remove(pre_processed_model)
    return quantized_model

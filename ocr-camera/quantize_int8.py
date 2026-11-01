"""fp32 ONNX -> int8 (weights of the LSTM and linear layers; the CNN stays fp32). 16.5 MB -> 7.5 MB, same accuracy.
usage: python quantize_int8.py model/crnn.onnx static-app/model/crnn_int8.onnx   (needs: pip install onnx onnxruntime)"""
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(sys.argv[1], sys.argv[2], weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM", "MatMul", "Gemm"])

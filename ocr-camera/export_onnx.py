"""Usage: python export_onnx.py [checkpoint.pt] [out.onnx]   (needs onnx: run with PYTHONPATH=deploy/pylib)"""
import sys, torch
from ocr_model import build_torch_model, IMG_H, IMG_W, CHARS
OUT = sys.argv[2] if len(sys.argv) > 2 else "deploy/crnn.onnx"
ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "data/crnn_final_real_only.pt", map_location="cpu")
assert ck["chars"] == CHARS and tuple(ck["img_hw"]) == (IMG_H, IMG_W), "checkpoint does not match ocr_model.py"
m = build_torch_model(); m.load_state_dict(ck["model"]); m.eval()
torch.onnx.export(m, torch.zeros(1, 1, IMG_H, IMG_W), OUT, input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "b"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
print("wrote", OUT)

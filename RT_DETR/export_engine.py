"""Export RT-DETR .pt checkpoint to an optimised inference format.

Supported formats (--format):
  onnx      ONNX graph  (.onnx)  — run with onnxruntime-gpu on Windows
  openvino  OpenVINO IR (.xml)   — run with Intel OpenVINO runtime
  engine    TensorRT    (.engine) — Linux / NVIDIA only

Install dependencies for each format:
  onnx:      pip install onnx onnxruntime-gpu
  openvino:  pip install openvino
  engine:    requires TensorRT (Linux / WSL2 only)
"""

import argparse
import os
from ultralytics import RTDETR

DEFAULT_PT_PATH = "./pdx_finetuned_rtdetr.pt"

EXTENSIONS = {
    "onnx": ".onnx",
    "openvino": "_openvino_model",
    "engine": ".engine",
}


def main():
    parser = argparse.ArgumentParser(description="Export RT-DETR to an optimised inference format.")
    parser.add_argument("--model", "-m", default=DEFAULT_PT_PATH, help="Input .pt checkpoint path.")
    parser.add_argument("--format", "-f", default="onnx",
                        choices=["onnx", "openvino", "engine"],
                        help="Export format (default: onnx).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--half", action="store_true", default=True,
                        help="Export with FP16 precision (ONNX/TRT; ignored for OpenVINO).")
    parser.add_argument("--batch", type=int, default=1, help="Max batch size (TensorRT only).")
    parser.add_argument("--dynamic", action="store_true", default=False,
                        help="Dynamic input axes for ONNX (needed if batch size varies).")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset version (default: 13; opset 14+ may break Squeeze on CUDAExecutionProvider).")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    # OpenVINO doesn't support FP16 export via the half flag
    half = args.half and args.format != "openvino"

    print(f"Exporting {args.model} -> format={args.format}, half={half}, imgsz={args.imgsz}")
    model = RTDETR(args.model)

    export_kwargs = dict(format=args.format, imgsz=args.imgsz, half=half)
    if args.format == "engine":
        export_kwargs["batch"] = args.batch
    if args.format == "onnx":
        export_kwargs["dynamic"] = args.dynamic
        export_kwargs["opset"] = args.opset

    model.export(**export_kwargs)

    base = os.path.splitext(args.model)[0]
    out = base + EXTENSIONS[args.format]
    print(f"\nDone. Exported model: {out}")
    print("Run inference with:")
    print(f"  python deepSORT_rtdetr.py --model {out}")

    if args.format == "onnx":
        print("\nMake sure onnxruntime-gpu is installed:")
        print("  pip install onnxruntime-gpu")


if __name__ == "__main__":
    main()

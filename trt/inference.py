import os
import argparse
import numpy as np
from pathlib import Path

from .engines.pytorch_engine import PyTorchEngine
from .engines.tensorrt_engine import TensorRTEngine
from .engines.onnx_tensorrt_engine import ONNXTensorRTEngine
from .utils import prepare_data, visualize_and_save_segmentation_result


def parse_args():
    parser = argparse.ArgumentParser(description="Run SwiftNet inference")
    parser.add_argument("--engine", type=str, choices=["pytorch", "tensorrt", "onnx-tensorrt"], default="pytorch")
    parser.add_argument("--weights", type=str, default="weights/rn18_single_scale/model_best.pt")
    parser.add_argument("--onnx_dir", type=str, default="trt/onnx", help="Directory with exported ONNX models")
    parser.add_argument("--dataset_path", type=str, default="datasets/cityscapes")
    parser.add_argument("--output", type=str, default="inference_output.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--engine_cache_dir", type=str, default="trt/engine_cache")
    return parser.parse_args()


def main():
    args = parse_args()
    image_size = (args.height, args.width)
    input_shape = (1, 3, args.height, args.width)

    # === Automatski generiraj ime ONNX i PLAN fajlova
    resolution_str = f"{args.height}x{args.width}"
    onnx_path = os.path.join(args.onnx_dir, f"trt_model_{resolution_str}.onnx")
    plan_path = os.path.join(args.engine_cache_dir, f"swiftnet_{resolution_str}.plan")

    if args.engine == "pytorch":
        engine = PyTorchEngine()
        engine.load_model(args.weights, image_size=image_size)
    elif args.engine == "tensorrt":
        engine, _ = TensorRTEngine.load_or_build(onnx_path, plan_path, input_shape)
    elif args.engine == "onnx-tensorrt":
        engine = ONNXTensorRTEngine()
        engine.load_model(onnx_path)
    else:
        raise ValueError("Unsupported engine")

    print(f"[INFO] Using engine: {engine.name}")
    print(f"[INFO] Input size: {args.height}x{args.width}")

    data_loader = prepare_data(args.dataset_path, subset="val", num_images=1, image_size=image_size)
    batch = next(iter(data_loader))
    input_data = engine.prepare_input(batch['image'])
    output = engine.run(input_data)
    logits = engine.get_logits_from_output(output)
    pred = np.argmax(logits[0], axis=0).astype(np.uint8)

    visualize_and_save_segmentation_result(pred, args.output)
    print(f"[INFO] Saved output to {args.output}")


if __name__ == "__main__":
    main()

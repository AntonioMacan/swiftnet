import argparse
import numpy as np
from .utils import prepare_data, visualize_and_save_segmentation_result

from .engines.pytorch_engine import PyTorchEngine
from .engines.tensorrt_engine import TensorRTEngine
from .engines.onnx_tensorrt_engine import ONNXTensorRTEngine


def parse_args():
    parser = argparse.ArgumentParser(description="SwiftNet Inference")
    parser.add_argument("--engine", type=str, choices=["pytorch", "tensorrt", "tensorrt-onnx"], default="pytorch")
    parser.add_argument("--weights", type=str, default="weights/rn18_single_scale/model_best.pt", help="Path to PyTorch weights")
    parser.add_argument("--onnx", type=str, default="trt/trt_model.onnx", help="Path to ONNX model (for TensorRT)")
    parser.add_argument("--dataset_path", type=str, default="datasets/cityscapes")
    parser.add_argument("--output", type=str, default="inference_result_swiftnet.png")
    return parser.parse_args()


def main():
    args = parse_args()

    # === Load appropriate engine
    if args.engine == "pytorch":
        engine = PyTorchEngine()
        engine.load_model(args.weights)
    elif args.engine == "tensorrt":
        engine = TensorRTEngine()
        engine.load_model(args.onnx)
    elif args.engine == "tensorrt-onnx":
        engine = ONNXTensorRTEngine()
        engine.load_model(args.onnx)
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")

    print(f"[INFO] Using engine: {engine.name}")

    # === Load data
    data_loader = prepare_data(args.dataset_path, subset='val', num_images=1)
    batch = next(iter(data_loader))
    input_data = engine.prepare_input(batch['image'])

    # === Run inference
    output = engine.run(input_data)
    logits = engine.get_logits_from_output(output)
    pred = np.argmax(logits[0], axis=0).astype(np.uint8)

    # === Save visualization
    visualize_and_save_segmentation_result(pred, args.output)
    print(f"[INFO] Saved segmentation result to {args.output}")


if __name__ == "__main__":
    main()

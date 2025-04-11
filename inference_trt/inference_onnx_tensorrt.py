import argparse
import time
import numpy as np
import onnx
import onnx_tensorrt.backend as backend
from .utils import (
    load_and_preprocess_image, 
    visualize_and_save_segmentation_result
)


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT inference using onnx_tensorrt')
    parser.add_argument('--onnx', 
                        type=str, 
                        default='inference_trt/trt_model.onnx',
                        help='Path to the ONNX model')
    parser.add_argument('--image', 
                        type=str,         
                        default='datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png',
                        help='Path to the input image')
    parser.add_argument('--output', 
                        type=str, 
                        default='inference_trt/inference_result_onnx_tensorrt.png',
                        help='Path to save the segmentation output')
    return parser.parse_args()


def build_engine_onnx(onnx_path):
    onnx_model = onnx.load(onnx_path)
    start_time = time.time()
    engine = backend.prepare(onnx_model, device="CUDA:0")
    engine_build_time = time.time() - start_time
    return engine, engine_build_time


def main():
    args = parse_args()
    
    # Create TensorRT engine
    print("[INFO] Building TensorRT engine...")
    engine, build_time = build_engine_onnx(args.onnx)
    print(f"[INFO] TensorRT engine built in {build_time:.2f} seconds")
    
    # Load and preprocess image
    print(f"[INFO] Processing image: {args.image}")
    image_tensor = load_and_preprocess_image(args.image)
    
    # Run single inference
    print("[INFO] Running inference...")
    start_time = time.time()
    output = engine.run(image_tensor)
    inference_time = time.time() - start_time
    print(f"[INFO] Inference completed in {inference_time*1000:.2f} ms")
    
    # Process output
    logits = output[0]
    pred = np.argmax(logits[0], axis=0).astype(np.uint8)
    
    visualize_and_save_segmentation_result(pred, args.output)
    print(f"[INFO] Saved segmentation result to: {args.output}")


if __name__ == "__main__":
    main()

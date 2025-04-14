import argparse
import time
import numpy as np
import onnx
import onnx_tensorrt.backend as backend
import torch
from time import perf_counter
from .utils import prepare_data
from .inference_onnx_tensorrt import build_engine_onnx


def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorRT performance measurement using onnx_tensorrt backend"
    )
    parser.add_argument('--onnx', 
                        type=str, 
                        default='inference_trt/trt_model.onnx',
                        help='Path to the ONNX model.')
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='datasets/cityscapes',
                        help='Path to the Cityscapes dataset.')
    parser.add_argument('--subset', 
                        type=str,
                        default='val',
                        help='Dataset subset (train, val).')
    parser.add_argument('--num_images', 
                        type=int, 
                        default=230,
                        help='Number of images to use from dataset.')
    parser.add_argument('--warmup', 
                        type=int, 
                        default=50,
                        help='Number of warm-up inferences (not measured).')
    parser.add_argument('--measure', 
                        type=int, 
                        default=180,
                        help='Number of inferences to measure.')
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Building TensorRT engine from ONNX...")
    engine, engine_build_time = build_engine_onnx(args.onnx)
    print(f"[INFO] TensorRT engine built in {engine_build_time:.2f} seconds")

    # Prepare dataset
    print(f"[INFO] Loading dataset from {args.dataset_path}...")
    data_loader = prepare_data(args.dataset_path, args.subset, args.num_images)
    print(f"[INFO] Loaded {len(data_loader)} images")

    # Warm-up phase
    print(f"[INFO] Running {args.warmup} warm-up inferences (not measured).")
    loader_iter = iter(data_loader)
    with torch.no_grad():
        for i in range(args.warmup):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                batch = next(loader_iter)
            input_data = batch['image'].numpy().astype(np.float32)
            _ = engine.run(input_data)
    torch.cuda.synchronize()

    # Measurement phase
    n = args.measure
    print(f"[INFO] Measuring time for {n} inferences...")
    
    # Reset iterator if needed
    loader_iter = iter(data_loader)

    times = []
    with torch.no_grad():
        for i in range(n):
            # Get next image (with cycling if needed)
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                batch = next(loader_iter)
            
            # Prepare input data
            input_data = batch['image'].numpy().astype(np.float32)
            
            # Measure inference time
            start_time = perf_counter()
            _ = engine.run(input_data)
            end_time = perf_counter()
            times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    mean_time = times.mean()
    fps = 1.0 / mean_time
    
    print(f"[RESULT] Total time: {times.sum():.2f} s")
    print(f"[RESULT] Mean inference time: {mean_time*1000:.2f} ms")
    print(f"[RESULT] Processing speed: {fps:.2f} FPS")

if __name__ == "__main__":
    main()

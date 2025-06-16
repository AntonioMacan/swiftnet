import argparse
import numpy as np
from time import perf_counter
from .utils import prepare_data

from .engines.pytorch_engine import PyTorchEngine
from .engines.tensorrt_engine import TensorRTEngine
from .engines.onnx_tensorrt_engine import ONNXTensorRTEngine


def parse_args():
    parser = argparse.ArgumentParser(description="SwiftNet Benchmark")
    parser.add_argument("--engine", type=str, choices=["pytorch", "tensorrt", "onnx-tensorrt"], default="pytorch")
    parser.add_argument("--weights", type=str, default="weights/rn18_single_scale/model_best.pt")
    parser.add_argument("--onnx", type=str, default="trt/trt_model.onnx")
    parser.add_argument("--dataset_path", type=str, default="datasets/cityscapes")
    parser.add_argument("--num_images", type=int, default=230)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=180)
    return parser.parse_args()


def run_benchmark(engine, data_loader, n_warmup=50, n_inference=180):
    loader_iter = iter(data_loader)

    print(f"[INFO] Warming up ({n_warmup} iterations)...")
    for _ in range(n_warmup):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(data_loader)
            batch = next(loader_iter)
        input_data = engine.prepare_input(batch['image'])
        _ = engine.run(input_data)
    if hasattr(input_data, "device") and input_data.device.type == "cuda":
        import torch
        torch.cuda.synchronize()

    print(f"[INFO] Benchmarking {n_inference} iterations...")
    times = []
    for _ in range(n_inference):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(data_loader)
            batch = next(loader_iter)
        input_data = engine.prepare_input(batch['image'])
        start = perf_counter()
        _ = engine.run(input_data)
        if hasattr(input_data, "device") and input_data.device.type == "cuda":
            torch.cuda.synchronize()
        end = perf_counter()
        times.append(end - start)

    times = np.array(times)
    print(f"[RESULT] Total time: {times.sum():.2f}s")
    print(f"[RESULT] Mean time: {times.mean()*1000:.2f}ms")
    print(f"[RESULT] FPS: {1.0 / times.mean():.2f}")


def main():
    args = parse_args()

    if args.engine == "pytorch":
        engine = PyTorchEngine()
        engine.load_model(args.weights)
    elif args.engine == "tensorrt":
        engine = TensorRTEngine()
        engine.load_model(args.onnx)
    elif args.engine == "onnx-tensorrt":
        engine = ONNXTensorRTEngine()
        engine.load_model(args.onnx)
    else:
        raise ValueError("Unsupported engine.")

    print(f"[INFO] Using engine: {engine.name}")
    data_loader = prepare_data(args.dataset_path, subset='val', num_images=args.num_images)
    run_benchmark(engine, data_loader, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
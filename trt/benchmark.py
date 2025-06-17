import os
import argparse
import numpy as np
from time import perf_counter

from .engines.pytorch_engine import PyTorchEngine
from .engines.tensorrt_engine import TensorRTEngine
from .engines.onnx_tensorrt_engine import ONNXTensorRTEngine
from .utils import prepare_data

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SwiftNet inference")
    parser.add_argument("--engine", type=str, choices=["pytorch", "tensorrt", "onnx-tensorrt"], default="pytorch")
    parser.add_argument("--weights", type=str, default="weights/rn18_single_scale/model_best.pt")
    parser.add_argument("--onnx", type=str, default="trt/trt_model.onnx")
    parser.add_argument("--dataset_path", type=str, default="datasets/cityscapes")
    parser.add_argument("--num_images", type=int, default=230)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--engine_cache_dir", type=str, default="trt/engine_cache")
    parser.add_argument("--resolutions", type=str, nargs="+", default=["1024x2048", "512x1024", "256x512"])
    return parser.parse_args()

def run_benchmark(engine, data_loader, n_warmup=50, n_inference=180):
    loader_iter = iter(data_loader)

    print(f"[INFO] Warming up ({n_warmup} iters)...")
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

    print(f"[INFO] Measuring {n_inference} iters...")
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
            import torch
            torch.cuda.synchronize()
        end = perf_counter()
        times.append(end - start)

    times = np.array(times)
    return {
        "mean_time": times.mean(),
        "fps": 1.0 / times.mean(),
        "total_time": times.sum()
    }

def parse_resolution(res_str):
    h, w = map(int, res_str.lower().split("x"))
    return (h, w)

def main():
    args = parse_args()
    results = []

    for res in args.resolutions:
        height, width = parse_resolution(res)
        print(f"\n=== Benchmarking at {height}x{width} ===")

        if args.engine == "pytorch":
            engine = PyTorchEngine()
            engine.load_model(args.weights, image_size=(height, width))
        elif args.engine == "tensorrt":
            onnx_path = f"trt/onnx/trt_model_{height}x{width}.onnx"
            cache_path = os.path.join(args.engine_cache_dir, f"swiftnet_{height}x{width}.plan")
            input_shape = (1, 3, height, width)
            engine, build_time = TensorRTEngine.load_or_build(onnx_path, cache_path, input_shape)
        elif args.engine == "onnx-tensorrt":
            engine = ONNXTensorRTEngine()
            engine.load_model(f"trt/onnx/trt_model_{height}x{width}.onnx")
        else:
            raise ValueError("Invalid engine selected")

        data_loader = prepare_data(
            args.dataset_path, subset="val",
            num_images=args.num_images,
            image_size=(height, width)
        )

        result = run_benchmark(engine, data_loader, args.warmup, args.iterations)
        result['resolution'] = f"{height}x{width}"
        results.append(result)

    for r in results:
        print(f"\n[RESULT] Resolution: {r['resolution']}")
        print(f"  Mean time: {r['mean_time']*1000:.2f} ms")
        print(f"  FPS: {r['fps']:.2f}")
        print(f"  Total time: {r['total_time']:.2f} s")

if __name__ == "__main__":
    main()

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


def compute_pixel_accuracy(pred1, pred2):
    assert pred1.shape == pred2.shape, "Shape mismatch"
    return np.mean(pred1 == pred2)


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

        image_size = (height, width)
        input_shape = (1, 3, height, width)
        resolution_str = f"{height}x{width}"

        # === Load inference engine ===
        if args.engine == "pytorch":
            engine = PyTorchEngine()
            engine.load_model(args.weights, image_size=image_size)
        elif args.engine == "tensorrt":
            onnx_path = f"trt/onnx/trt_model_{resolution_str}.onnx"
            cache_path = os.path.join(args.engine_cache_dir, f"swiftnet_{resolution_str}.plan")
            engine, build_time = TensorRTEngine.load_or_build(onnx_path, cache_path, input_shape)
        elif args.engine == "onnx-tensorrt":
            engine = ONNXTensorRTEngine()
            engine.load_model(f"trt/onnx/trt_model_{resolution_str}.onnx")
        else:
            raise ValueError("Invalid engine selected")

        data_loader = prepare_data(
            args.dataset_path, subset="val",
            num_images=args.num_images,
            image_size=image_size
        )

        result = run_benchmark(engine, data_loader, args.warmup, args.iterations)
        result['resolution'] = resolution_str

        # === Pixel-wise accuracy (ako nije PyTorch engine) ===
        if args.engine != "pytorch":
            print("[INFO] Computing pixel-wise accuracy vs PyTorch...")

            # Pokreni PyTorch model
            torch_engine = PyTorchEngine()
            torch_engine.load_model(args.weights, image_size=image_size)

            torch_loader = prepare_data(
                args.dataset_path, subset="val",
                num_images=5,
                image_size=image_size
            )

            batch = next(iter(torch_loader))
            input_torch = torch_engine.prepare_input(batch['image'])
            out_torch = torch_engine.run(input_torch)
            preds_torch = [np.argmax(t, axis=0) for t in torch_engine.get_logits_from_output(out_torch)]

            input_target = engine.prepare_input(batch['image'])
            out_target = engine.run(input_target)
            preds_target = [np.argmax(t, axis=0) for t in engine.get_logits_from_output(out_target)]

            pixel_accs = [
                compute_pixel_accuracy(p1, p2)
                for p1, p2 in zip(preds_torch, preds_target)
            ]
            pixel_accuracy = sum(pixel_accs) / len(pixel_accs)
            result["pixel_accuracy"] = pixel_accuracy

        results.append(result)

    for r in results:
        print(f"\n[RESULT] Resolution: {r['resolution']}")
        print(f"  Mean time: {r['mean_time']*1000:.2f} ms")
        print(f"  FPS: {r['fps']:.2f}")
        print(f"  Total time: {r['total_time']:.2f} s")
        if "pixel_accuracy" in r:
            print(f"  Pixel-wise accuracy vs PyTorch: {r['pixel_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()

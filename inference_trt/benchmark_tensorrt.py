import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # init CUDA context
from time import perf_counter
from .inference_tensorrt import build_engine_onnx
from .utils import prepare_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorRT performance measurement using tensorrt"
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
                        default=100,
                        help='Number of images to use from dataset.')
    parser.add_argument('--warmup', 
                        type=int, 
                        default=20,
                        help='Number of warm-up inferences (not measured).')
    parser.add_argument('--measure', 
                        type=int, 
                        default=100,
                        help='Number of inferences to measure.')
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Building TensorRT engine from ONNX...")
    engine, engine_build_time = build_engine_onnx(args.onnx)
    print(f"[INFO] TensorRT engine built in {engine_build_time:.2f} seconds")

    # Create an execution context and a CUDA stream
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Load data
    print(f"[INFO] Loading dataset from {args.dataset_path}...")
    data_loader = prepare_data(args.dataset_path, args.subset, args.num_images)
    print(f"[INFO] Loaded {len(data_loader)} images")

    # Model output shape for Cityscapes
    output_shape = (1, 19, 1024, 2048)
    output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

    # Allocate memory for output
    d_output = cuda.mem_alloc(output_size)
    h_output = cuda.pagelocked_empty(shape=output_shape, dtype=np.float32)

    # Warm-up phase
    print(f"[INFO] Running {args.warmup} warm-up inferences (not measured).")
    loader_iter = iter(data_loader)
    for _ in range(min(args.warmup, len(data_loader))):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(data_loader)
            batch = next(loader_iter)

        input_data = batch['image'].numpy().astype(np.float32)
        input_size = input_data.nbytes
        d_input = cuda.mem_alloc(input_size)

        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        d_input.free()  # Free memory after each iteration

    # Measurement phase
    n = args.measure
    print(f"[INFO] Measuring time for {n} inferences...")

    # Reset iterator if needed
    loader_iter = iter(data_loader)

    times = []
    for i in range(n):
        # Get next image (with cycling if needed)
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(data_loader)
            batch = next(loader_iter)

        # Prepare input data
        input_data = batch['image'].numpy().astype(np.float32)
        input_size = input_data.nbytes
        d_input = cuda.mem_alloc(input_size)

        # Measure inference time
        start_time = perf_counter()
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        end_time = perf_counter()

        times.append(end_time - start_time)
        d_input.free()  # Free memory after each iteration
    
    # Calculate statistics
    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()
    fps = 1.0 / mean_time
    
    print(f"[RESULT] Total time: {times.sum():.2f} s")
    print(f"[RESULT] Mean inference time: {mean_time*1000:.2f} ms")
    print(f"[RESULT] Processing speed: {fps:.2f} FPS")


if __name__ == "__main__":
    main()

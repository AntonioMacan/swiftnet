import argparse
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # init CUDA context
from time import perf_counter
from .utils import load_and_preprocess_image
from .inference_tensorrt import build_engine_onnx


def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorRT performance measurement"
    )
    parser.add_argument('--onnx', 
                        type=str, 
                        default='inference_trt/trt_model.onnx',
                        help='Path to the ONNX model.')
    parser.add_argument('--image', 
                        type=str, 
                        default='datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png',
                        help='Path to an input Cityscapes image.')
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

    # Load and preprocess the input image
    print(f"[INFO] Loading and preprocessing image: {args.image}")
    input_data = load_and_preprocess_image(args.image)
    input_size = input_data.nbytes

    # Model output shape for Cityscapes
    output_shape = (1, 19, 1024, 2048)
    output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

    # Allocate memory on the device (GPU)
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    # Pinned memory for outputs
    h_output = cuda.pagelocked_empty(shape=output_shape, dtype=np.float32)

    # Warm-up
    print(f"[INFO] Running {args.warmup} warm-up inferences (not measured).")
    for _ in range(args.warmup):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Measurement
    n = args.measure
    print(f"[INFO] Measuring time for {n} inferences...")
    start_time = perf_counter()
    for _ in range(n):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    end_time = perf_counter()
    fps = n / (end_time - start_time)

    print(f"[RESULT] Processing speed: {fps:.2f} FPS")


if __name__ == "__main__":
    main()

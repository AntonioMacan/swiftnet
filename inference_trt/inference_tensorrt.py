import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # init CUDA context
import time
import argparse
from .utils import (
    load_and_preprocess_image, 
    visualize_and_save_segmentation_result
)


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT inference using tensorrt')
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
                        default='inference_trt/inference_result_tensorrt.png',
                        help='Path to save the segmentation output')
    return parser.parse_args()


def build_engine_onnx(onnx_path, workspace_size=1<<30):
    """Build a TensorRT engine from the given ONNX file"""
    
    start_time = time.time()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as model_file:
        model_data = model_file.read()
    if not parser.parse(model_data):
        raise RuntimeError("Parser is not able to load ONNX model.")
    
    # Configure engine build
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # Build a serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        raise RuntimeError("Failed to build a serialzed TensorRT network!")
    
    # Deserialize it into an engine
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    engine_build_time = time.time() - start_time
    if engine is None:
        engine_build_time = None
        raise RuntimeError("Failed to build TensorRT engine.")
    
    return engine, engine_build_time


def main():
    args = parse_args()

    print("[INFO] Building TensorRT engine from ONNX...")
    engine, engine_build_time = build_engine_onnx(args.onnx)
    print(f"[INFO]TensorRT engine built in {engine_build_time:.2f} seconds")

    # Create an execution context and a CUDA stream
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Load and preprocess the input image
    print(f"[INFO] Loading and preprocessing image: {args.image}")
    input_data = load_and_preprocess_image(args.image)
    input_size = input_data.nbytes

    # Model output for Cityscapes
    output_shape = (1, 19, 1024, 2048)
    output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

    # Allocate memory on the device (GPU)
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    # Pinned memory for outputs
    h_output = cuda.pagelocked_empty(shape=output_shape, dtype=np.float32)

    print("[INFO] Running inference...")
    start_time = time.time()
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    elapsed = time.time() - start_time
    print(f"[INFO] Inference time: {elapsed*1000:.2f} ms")
    
    # Process output
    logits = h_output.reshape((19, 1024, 2048))
    pred = np.argmax(logits, axis=0).astype(np.uint8)
    visualize_and_save_segmentation_result(pred, args.output)
    print(f"[INFO] Saved segmentation result to: {args.output}")


if __name__ == '__main__':
    main()

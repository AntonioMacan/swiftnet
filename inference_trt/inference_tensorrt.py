import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # init CUDA context
from PIL import Image
import time
import argparse

from torchvision.transforms import Compose
from data.transform import Open, Normalize, Tensor, ColorizeLabels
from data.cityscapes import Cityscapes


# Cityscapes related
scale = 255
mean = Cityscapes.mean
std = Cityscapes.std
color_info = Cityscapes.color_info
to_color = ColorizeLabels(color_info)

def load_and_prepare_image(image_path):
    transforms = Compose([
        Open(),
        Normalize(scale=scale, mean=mean, std=std),
        Tensor()
    ])
    
    # Load image
    sample = {"image": image_path}
    sample = transforms(sample)
    
    # Get preprocessed image tensor
    image_tensor = sample["image"].unsqueeze(0).numpy().astype(np.float32)
    return image_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='inference_trt/trt_model.onnx')
    parser.add_argument('--image', type=str, default='datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png')
    parser.add_argument('--output', type=str, default='inference_trt/inference_result_tensorrt.png')
    args = parser.parse_args()

    # Create TensorRT engine
    print("Building TensorRT engine...")
    start_time = time.time()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, 'rb') as model:
        parser.parse(model.read())

    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    build_time = time.time() - start_time
    print(f"TensorRT engine built in {build_time:.2f} seconds")

    # Load and preprocess image
    print("Preprocessing image...")
    input_data = load_and_prepare_image(args.image)

    context = engine.create_execution_context()
    d_input = cuda.mem_alloc(1 * input_data.nbytes)
    output_shape = (19, 1024, 2048)
    output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)
    d_output = cuda.mem_alloc(output_size)

    stream = cuda.Stream()

    output_data = cuda.pagelocked_empty(int(np.prod(output_shape)), dtype=np.float32).reshape(output_shape)

    # Run single inference
    print("Running inference...")
    start_time = time.time()
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()
    elapsed = time.time() - start_time
    print(f"Inference time: {elapsed*1000:.2f} ms")

    # Process output
    logits = output_data.reshape((19, 1024, 2048))
    pred = np.argmax(logits, axis=0).astype(np.uint8)

    # Visualize and save segmentation result
    colored_pred = to_color(pred)
    colored_pred_img = Image.fromarray(colored_pred)
    colored_pred_img.save(args.output)
    print(f"Saved segmentation to: {args.output}")


if __name__ == '__main__':
    main()

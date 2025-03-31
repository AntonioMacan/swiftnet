import argparse
import time
import numpy as np
import onnx
import onnx_tensorrt.backend as backend
from PIL import Image
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
    image_tensor = sample["image"].unsqueeze(0).numpy()
    return image_tensor

def main():
    parser = argparse.ArgumentParser(description='TensorRT inference using onnx_tensorrt')
    parser.add_argument('--onnx', type=str, default='inference_trt/trt_model.onnx',
                        help='Path to the ONNX model')
    parser.add_argument('--image', type=str,         
                        default='datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png',
                        help='Path to the input image')
    parser.add_argument('--output', type=str, default='inference_trt/inference_result_onnx_tensorrt.png',
                        help='Path to save the segmentation output')
    args = parser.parse_args()
    
    # Load ONNX model
    print(f"Loading ONNX model from {args.onnx}...")
    onnx_model = onnx.load(args.onnx)
    
    # Create TensorRT engine
    print("Building TensorRT engine...")
    start_time = time.time()
    engine = backend.prepare(onnx_model, device="CUDA:0")
    engine_build_time = time.time() - start_time
    print(f"TensorRT engine built in {engine_build_time:.2f} seconds")
    
    # Load and preprocess image
    print(f"Processing image: {args.image}")
    image_tensor = load_and_prepare_image(args.image)
    
    # Run single inference
    print("Running inference...")
    start_time = time.time()
    output = engine.run(image_tensor)
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time*1000:.2f} ms")
    
    # Process output
    logits = output[0]
    pred = np.argmax(logits[0], axis=0).astype(np.uint8)
    
    # Visualize and save segmentation result
    colored_pred = to_color(pred)
    colored_pred_img = Image.fromarray(colored_pred)
    colored_pred_img.save(args.output)
    print(f"Saved segmentation result to: {args.output}")


if __name__ == "__main__":
    main()

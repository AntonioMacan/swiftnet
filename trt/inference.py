import os
import argparse
import numpy as np
from pathlib import Path

from .engines.pytorch_engine import PyTorchEngine
from .engines.tensorrt_engine import TensorRTEngine
from .engines.onnx_tensorrt_engine import ONNXTensorRTEngine
from .utils import prepare_data, save_segmentation_result, save_comparison_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Run SwiftNet inference")
    parser.add_argument("--engine", type=str, choices=["pytorch", "tensorrt", "onnx-tensorrt"], default="pytorch")
    parser.add_argument("--weights", type=str, default="weights/rn18_single_scale/model_best.pt")
    parser.add_argument("--onnx_dir", type=str, default="trt/onnx", help="Directory with exported ONNX models")
    parser.add_argument("--dataset_path", type=str, default="datasets/cityscapes")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trt/results",
        help="Directory where results will be written, preserving Cityscapes folder structure",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of images to run inference on (all if None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--engine_cache_dir", type=str, default="trt/engine_cache")
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save side-by-side comparison of original image and segmentation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_size = (args.height, args.width)
    input_shape = (1, 3, args.height, args.width)

    out_root = Path(args.output_dir)
    # Create output directory if it doesn't exist
    out_root.mkdir(parents=True, exist_ok=True)

    # === Automatski generiraj ime ONNX i PLAN fajlova
    resolution_str = f"{args.height}x{args.width}"
    onnx_path = os.path.join(args.onnx_dir, f"trt_model_{resolution_str}.onnx")
    plan_path = os.path.join(args.engine_cache_dir, f"swiftnet_{resolution_str}.plan")

    if args.engine == "pytorch":
        engine = PyTorchEngine()
        build_time = engine.load_model(args.weights, image_size=image_size)
        print(f"[INFO] Using {engine.name} engine")
        print(f"[INFO] Built in {build_time:.2f}s")
    elif args.engine == "tensorrt":
        engine, build_time = TensorRTEngine.load_or_build(onnx_path, plan_path, input_shape)
        print(f"[INFO] Using {engine.name} engine")
        print(f"[INFO] {'Loaded from cache' if build_time == 0 else f'Built in {build_time:.2f}s'}")
    elif args.engine == "onnx-tensorrt":
        engine = ONNXTensorRTEngine()
        build_time = engine.load_model(onnx_path)
        print(f"[INFO] Using {engine.name} engine")
        print(f"[INFO] Built in {build_time:.2f}s")
    else:
        raise ValueError("Unsupported engine")

    print(f"[INFO] Input size: {args.height}x{args.width}")

    data_loader = prepare_data(
        args.dataset_path, 
        subset="val", 
        num_images=args.num_images, 
        batch_size=args.batch_size,
        image_size=image_size
    )

    dataset_root = Path(args.dataset_path)

    for batch in data_loader:
        # Extract images and paths from batch
        images = batch['image']  # This should be a tensor from SwiftNet dataloader
        paths = batch['name']  # SwiftNet uses 'name' field for paths
        
        # Handle single image case (when batch_size=1, paths might be a string)
        if isinstance(paths, str):
            paths = [paths]
        elif not isinstance(paths, list):
            # Convert tensor or other iterable to list
            paths = list(paths)

        input_data = engine.prepare_input(images)
        output = engine.run(input_data)
        logits_batch = engine.get_logits_from_output(output)

        for i, (logits, img_name) in enumerate(zip(logits_batch, paths)):
            pred = np.argmax(logits, axis=0).astype(np.uint8)

            # Construct path from image name
            # img_name should be something like 'aachen_000000_000019_leftImg8bit'
            # We need to find the corresponding city and construct the path
            
            # Find the original image path in the dataset
            img_path = None
            subset_dir = dataset_root / 'leftImg8bit' / 'val'
            for city_dir in subset_dir.iterdir():
                if city_dir.is_dir():
                    potential_path = city_dir / f"{img_name}.png"
                    if potential_path.exists():
                        img_path = potential_path
                        break
            
            if img_path is None:
                print(f"[WARNING] Could not find original image for {img_name}")
                continue

            # Compute relative path w.r.t. dataset root (keeps subset/city folders)
            rel_path = img_path.relative_to(dataset_root)

            pred_name = f"pred_{args.engine}_{args.height}x{args.width}{rel_path.suffix}"
            save_path = out_root / rel_path.parent / rel_path.stem / pred_name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_segmentation_result(pred, save_path)
            print(f"[INFO] Saved result: {save_path}")

            # Save comparison if requested
            if args.save_comparison:
                comparison_name = f"comparison_{args.engine}_{args.height}x{args.width}{rel_path.suffix}"
                comparison_path = out_root / rel_path.parent / rel_path.stem / comparison_name
                
                # Load original image for comparison
                from PIL import Image
                original_image = Image.open(img_path).convert('RGB')
                save_comparison_visualization(original_image, pred, comparison_path)
                print(f"[INFO] Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
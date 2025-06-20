import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from time import perf_counter
from torchmetrics import ConfusionMatrix
from pathlib import Path

from .engines.pytorch_engine import PyTorchEngine
from .engines.tensorrt_engine import TensorRTEngine
from .engines.onnx_tensorrt_engine import ONNXTensorRTEngine
from .utils import prepare_data

# Import evaluation functions from SwiftNet
from evaluation.evaluate import compute_errors


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SwiftNet inference")
    parser.add_argument("--engine", type=str, choices=["pytorch", "tensorrt", "onnx-tensorrt"], default="pytorch")
    parser.add_argument("--weights", type=str, default="weights/rn18_single_scale/model_best.pt")
    parser.add_argument("--onnx", type=str, default="trt/trt_model.onnx")
    parser.add_argument("--dataset_path", type=str, default="datasets/cityscapes")
    parser.add_argument("--num_images", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--engine_cache_dir", type=str, default="trt/engine_cache")
    parser.add_argument("--resolutions", type=str, nargs="+", default=["1024x2048", "512x1024", "256x512"])
    return parser.parse_args()


def prepare_evaluation_data(dataset_path, subset='val', num_images=None, image_size=(1024, 2048)):
    """Prepare data loader specifically for evaluation with ground truth labels"""
    from torchvision.transforms import Compose
    from torch.utils.data import DataLoader, Subset
    from data.transform import Open, RemapLabels, Normalize, Tensor, Resize
    from data.cityscapes import Cityscapes
    
    # Define transforms that resize input images but keep original ground truth size
    class ResizeImageOnly:
        def __init__(self, size):
            self.size = size  # (width, height) for PIL
            
        def __call__(self, example):
            if 'image' in example:
                from PIL import Image
                example['image'] = example['image'].resize(self.size, Image.BILINEAR)
            return example
    
    # Use transforms that don't change ground truth resolution
    trans_val = Compose([
        Open(),
        RemapLabels(Cityscapes.map_to_id, ignore_id=255, ignore_class=Cityscapes.num_classes),
        ResizeImageOnly((image_size[1], image_size[0])),  # Only resize input image
        Normalize(scale=255, mean=Cityscapes.mean, std=Cityscapes.std),
        Tensor(),
    ])
    
    dataset = Cityscapes(Path(dataset_path), transforms=trans_val, subset=subset)
    
    if num_images and num_images < len(dataset):
        indices = list(range(num_images))
        dataset = Subset(dataset, indices)
    
    # Import the custom collate function
    from data.transform.base import custom_collate
    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate)
    
    return loader, dataset


def evaluate_miou(engine, data_loader, class_info, ignore_id=19):
    """
    Evaluate mIoU using SwiftNet's evaluation methodology
    """
    conf_matrix = ConfusionMatrix(task="multiclass", num_classes=len(class_info))
    conf_mat = np.zeros((len(class_info), len(class_info)), dtype=np.uint64)
    
    print(f"[INFO] Evaluating mIoU on {len(data_loader)} images...")
    
    for step, batch in enumerate(data_loader):
        if step % 10 == 0:
            print(f"  Processing image {step}/{len(data_loader)}")
        
        # Get ground truth labels and their shape
        gt_labels = batch['original_labels'].int().cpu()
        gt_shape = gt_labels.shape[-2:]  # Get H, W of ground truth
        
        # Prepare input for engine
        input_data = engine.prepare_input(batch['image'])
        
        # Run inference
        output = engine.run(input_data)
        logits_batch = engine.get_logits_from_output(output)
        
        # Convert logits to predictions
        for logits, gt in zip(logits_batch, gt_labels):
            # Ensure logits is a tensor
            if not isinstance(logits, torch.Tensor):
                logits = torch.from_numpy(logits)
            
            # Check if upsampling is needed
            logits_shape = logits.shape[-2:]  # Get H, W of logits
            if logits_shape != gt_shape:
                print(f"  Upsampling logits from {logits_shape} to {gt_shape}")
                # Upsample logits to match ground truth size
                logits = F.interpolate(
                    logits.unsqueeze(0),  # Add batch dimension
                    size=gt_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # Remove batch dimension
            
            # Convert to predictions
            pred = torch.argmax(logits, dim=0).int().cpu()
            
            # Filter out ignore pixels
            if ignore_id != -100:
                valid_idx = gt != ignore_id
                pred_valid = pred[valid_idx]
                gt_valid = gt[valid_idx]
            else:
                pred_valid = pred.flatten()
                gt_valid = gt.flatten()
            
            # Accumulate confusion matrix
            conf_mat += conf_matrix(pred_valid, gt_valid).numpy().astype(np.uint64)
    
    # Compute metrics using SwiftNet's function
    pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(
        conf_mat, class_info, verbose=False
    )
    
    return iou_acc, per_class_iou, pixel_acc


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

        # Performance benchmark
        data_loader = prepare_data(
            args.dataset_path, subset="val",
            num_images=args.num_images,
            image_size=image_size
        )

        result = run_benchmark(engine, data_loader, args.warmup, args.iterations)
        result['resolution'] = resolution_str

        # === mIoU evaluation ===
        print("[INFO] Computing mIoU...")
        
        # Prepare evaluation data with ground truth
        eval_loader, eval_dataset = prepare_evaluation_data(
            args.dataset_path, subset="val",
            image_size=image_size
        )
        
        # Get class info from dataset
        class_info = eval_dataset.dataset.class_info if hasattr(eval_dataset, 'dataset') else eval_dataset.class_info
        
        # Evaluate mIoU for the current engine only
        miou, per_class_iou, pixel_acc = evaluate_miou(engine, eval_loader, class_info)
        result["miou"] = miou
        result["pixel_accuracy"] = pixel_acc

        # === Pixel agreement vs PyTorch ===
        if args.engine != "pytorch":
            print("[INFO] Comparing predictions with PyTorch baseline...")
            torch_engine = PyTorchEngine()
            torch_engine.load_model(args.weights, image_size=image_size)

            torch_loader = prepare_data(args.dataset_path, subset="val", image_size=image_size)
            batch = next(iter(torch_loader))
            input_torch = torch_engine.prepare_input(batch['image'])
            out_torch = torch_engine.run(input_torch)
            preds_torch = [np.argmax(t.cpu().numpy(), axis=0) for t in torch_engine.get_logits_from_output(out_torch)]

            input_target = engine.prepare_input(batch['image'])
            out_target = engine.run(input_target)
            logits_target = engine.get_logits_from_output(out_target)
            preds_target = [np.argmax(t.cpu().numpy(), axis=0) for t in logits_target]

            pixel_accs = [
                compute_pixel_accuracy(p1, p2)
                for p1, p2 in zip(preds_torch, preds_target)
            ]
            pixel_accuracy = sum(pixel_accs) / len(pixel_accs)
            result["engine_agreement"] = pixel_accuracy

        results.append(result)

    # Print results
    for r in results:
        print(f"\n[RESULT] Resolution: {r['resolution']}")
        print(f"  Mean time: {r['mean_time']*1000:.2f} ms")
        print(f"  FPS: {r['fps']:.2f}")
        print(f"  Total time: {r['total_time']:.2f} s")
        print(f"  mIoU: {r['miou']:.2f}%")
        if args.engine != "pytorch":
            print(f"  Pixel agreement vs PyTorch: {r['engine_agreement'] * 100:.2f}%")


if __name__ == "__main__":
    main()
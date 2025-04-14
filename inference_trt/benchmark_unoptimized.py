import argparse
import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter

from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import resnet18
from models.loss import SemsegCrossEntropy
from data.cityscapes import Cityscapes

from .utils import prepare_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch model performance benchmark"
    )
    parser.add_argument('--weights', 
                        type=str, 
                        default='weights/rn18_single_scale/model_best.pt',
                        help='Path to the model weights.')
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
                        default=230,
                        help='Number of images to use from dataset.')
    parser.add_argument('--warmup', 
                        type=int, 
                        default=50,
                        help='Number of warm-up inferences (not measured).')
    parser.add_argument('--measure', 
                        type=int, 
                        default=180,
                        help='Number of inferences to measure.')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Initialize the model
    print("[INFO] Initializing model...")
    num_classes = Cityscapes.num_classes
    resnet = resnet18(pretrained=True, efficient=False)
    model = SemsegModel(resnet, num_classes, k=3, bias=False)
    model.criterion = SemsegCrossEntropy(num_classes=num_classes, ignore_id=num_classes)
    
    # Load weights
    print(f"[INFO] Loading weights from {args.weights}...")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Prepare dataset
    print(f"[INFO] Loading dataset from {args.dataset_path}...")
    data_loader = prepare_data(args.dataset_path, args.subset, args.num_images)
    print(f"[INFO] Loaded {len(data_loader)} images")
    
    # Target dimensions for the model
    TARGET_SIZE = (1024 // 4, 2048 // 4)  # Feature map size = image size / 4
    IMAGE_SIZE = (1024, 2048)
    
    # Warm-up phase
    print(f"[INFO] Running {args.warmup} warm-up inferences (not measured).")
    loader_iter = iter(data_loader)
    with torch.no_grad():
        for i in range(min(args.warmup, len(data_loader))):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                batch = next(loader_iter)
            
            # Prepare input data
            image = batch['image'].to(device)
            
            # Run inference
            _, _ = model(image, TARGET_SIZE, IMAGE_SIZE)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measurement phase
    n = args.measure
    print(f"[INFO] Measuring time for {n} inferences...")
    
    # Reset iterator if needed
    loader_iter = iter(data_loader)
    
    times = []
    with torch.no_grad():
        for i in tqdm(range(n)):
            # Get next image (with cycling if needed)
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_loader)
                batch = next(loader_iter)
            
            # Prepare input data
            image = batch['image'].to(device)
            
            # Measure inference time
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = perf_counter()
            
            _, _ = model(image, TARGET_SIZE, IMAGE_SIZE)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = perf_counter()
            
            times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    mean_time = times.mean()
    fps = 1.0 / mean_time
    
    print(f"[RESULT] Total time: {times.sum():.2f} s")
    print(f"[RESULT] Mean inference time: {mean_time*1000:.2f} ms")
    print(f"[RESULT] Processing speed: {fps:.2f} FPS")


if __name__ == "__main__":
    main()

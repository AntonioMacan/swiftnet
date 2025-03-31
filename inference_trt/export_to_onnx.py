import torch
import argparse
from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import resnet18
from models.loss import SemsegCrossEntropy
from data.cityscapes import Cityscapes


# Input dimensions - Cityscapes full resolution
INPUT_SHAPE = (3, 1024, 2048)           # Original Cityscapes resolution
TARGET_SIZE = (1024 // 4, 2048 // 4)    # Feature map size = image size / 4
IMAGE_SIZE = (1024, 2048)

# Number of classes for Cityscapes dataset
num_classes = Cityscapes.num_classes    # if working with something other than Cityscapes, implement and import that class  # noqa

def main():
    parser = argparse.ArgumentParser(description='Export SwiftNet model to ONNX')
    parser.add_argument('--weights', type=str, default='weights/rn18_single_scale/model_best.pt',
                        help='Path to the model weights')
    parser.add_argument('--output', type=str, default='inference_trt/trt_model.onnx',
                        help='Path to save the ONNX model')
    args = parser.parse_args()

    # Initialize backbone and model
    print("Initializing ResNet backbone...")
    resnet = resnet18(pretrained=True, efficient=False)
    
    model = SemsegModel(resnet, num_classes, k=3, bias=False)
    model.upsample_logits = False
    model.criterion = SemsegCrossEntropy(num_classes=num_classes, ignore_id=num_classes)
    
    # Always use CUDA
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Load pretrained weights
    print(f"Loading weights from {args.weights}...")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Model loaded successfully.")
    
    # Create wrapper class for ONNX export
    class SwiftnetONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, image):
            logits, _ = self.model(image, TARGET_SIZE, IMAGE_SIZE)
            return logits
    
    # Wrap model for simplified ONNX export
    wrapped_model = SwiftnetONNXWrapper(model).to(device)
    
    # Dummy input
    print(f"Creating dummy input with shape: {INPUT_SHAPE}...")
    x = torch.randn(1, *INPUT_SHAPE).to(device)

    # Run a forward pass through the model
    print("Running a forward pass through the model...")
    torch.out = wrapped_model(x)

    # Export model to ONNX format
    print("Exporting model to ONNX format...")
    torch.onnx.export(
        wrapped_model,
        x,
        args.output,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    
    print(f"ONNX model has been saved as '{args.output}'")


if __name__ == "__main__":
    main()
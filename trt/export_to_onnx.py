import argparse
import torch
from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import resnet18
from models.loss import SemsegCrossEntropy
from data.cityscapes import Cityscapes
import os

NUM_CLASSES = Cityscapes.num_classes

class SwiftNetONNXWrapper(torch.nn.Module):
    def __init__(self, model, target_size, image_size):
        super().__init__()
        self.model = model
        self.target_size = target_size
        self.image_size = image_size

    def forward(self, x):
        logits, _ = self.model(x, self.target_size, self.image_size)
        return logits

def parse_args():
    parser = argparse.ArgumentParser(description="Export SwiftNet model to ONNX")
    parser.add_argument("--weights", type=str, default='weights/rn18_single_scale/model_best.pt', help="Path to PyTorch weights")
    parser.add_argument("--output_dir", type=str, default="trt/onnx", help="Directory to save ONNX models")
    parser.add_argument("--height", type=int, default=1024, help="Input height")
    parser.add_argument("--width", type=int, default=2048, help="Input width")
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.height, args.width)
    target_size = (args.height // 4, args.width // 4)
    input_shape = (3, args.height, args.width)

    # === Init model
    backbone = resnet18(pretrained=True, efficient=False)
    model = SemsegModel(backbone, NUM_CLASSES, k=3, bias=False)
    model.upsample_logits = False
    model.criterion = SemsegCrossEntropy(num_classes=NUM_CLASSES, ignore_id=NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval().to(device)

    wrapped_model = SwiftNetONNXWrapper(model, target_size, image_size).to(device)
    dummy_input = torch.randn(1, *input_shape).to(device)

    # === Output path s rezolucijom
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"trt_model_{args.height}x{args.width}.onnx")

    print(f"[INFO] Exporting ONNX to {output_path}")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

    print(f"[INFO] ONNX exported successfully to: {output_path}")

if __name__ == "__main__":
    main()

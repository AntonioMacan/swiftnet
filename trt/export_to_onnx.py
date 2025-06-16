import argparse
import torch
from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import resnet18
from models.loss import SemsegCrossEntropy
from data.cityscapes import Cityscapes

# Dimenzije za Cityscapes
INPUT_SHAPE = (3, 1024, 2048)
TARGET_SIZE = (1024 // 4, 2048 // 4)
IMAGE_SIZE = (1024, 2048)
NUM_CLASSES = Cityscapes.num_classes

class SwiftNetONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x, TARGET_SIZE, IMAGE_SIZE)
        return logits

def parse_args():
    parser = argparse.ArgumentParser(description="Export SwiftNet model to ONNX")
    parser.add_argument("--weights", type=str, default="weights/rn18_single_scale/model_best.pt")
    parser.add_argument("--output", type=str, default="trt/trt_model.onnx")
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # === Inicijalizacija modela
    print("[INFO] Initializing SwiftNet model...")
    backbone = resnet18(pretrained=True, efficient=False)
    model = SemsegModel(backbone, NUM_CLASSES, k=3, bias=False)
    model.upsample_logits = False
    model.criterion = SemsegCrossEntropy(num_classes=NUM_CLASSES, ignore_id=NUM_CLASSES)

    # === Učitavanje težina
    print(f"[INFO] Loading weights from {args.weights}...")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval().to(device)

    # === Omotavanje za ONNX export
    wrapped_model = SwiftNetONNXWrapper(model).to(device)

    # === Dummy input za tracing
    dummy_input = torch.randn(1, *INPUT_SHAPE).to(device)

    # === Export
    print(f"[INFO] Exporting to ONNX at: {args.output}")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

    print(f"[INFO] ONNX model exported successfully to: {args.output}")

if __name__ == "__main__":
    main()
import torch
import numpy as np
from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import resnet18
from models.loss import SemsegCrossEntropy
from data.cityscapes import Cityscapes
from .base_engine import InferenceEngine

TARGET_SIZE = (1024 // 4, 2048 // 4)
IMAGE_SIZE = (1024, 2048)
INPUT_SHAPE = (3, 1024, 2048)

class PyTorchEngine(InferenceEngine):
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self):
        return "SwiftNet-PyTorch"

    def load_model(self, model_path):
        num_classes = Cityscapes.num_classes
        resnet = resnet18(pretrained=True, efficient=False)
        model = SemsegModel(resnet, num_classes, k=3, bias=False)
        model.upsample_logits = False
        model.criterion = SemsegCrossEntropy(num_classes=num_classes, ignore_id=num_classes)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval().to(self.device)
        self.model = model
        return 0.0

    def prepare_input(self, images):
        return images.to(self.device)

    def run(self, input_data):
        with torch.no_grad():
            logits, _ = self.model(input_data, TARGET_SIZE, IMAGE_SIZE)
        return logits

    def get_logits_from_output(self, output):
        return [output[i].detach().cpu().numpy() for i in range(output.shape[0])]
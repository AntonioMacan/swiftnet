import torch
import numpy as np
from models.semseg import SemsegModel
from models.resnet.resnet_single_scale import resnet18
from models.loss import SemsegCrossEntropy
from data.cityscapes import Cityscapes
from .base_engine import InferenceEngine

NUM_CLASSES = Cityscapes.num_classes

class PyTorchEngine(InferenceEngine):
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = None
        self.image_size = None

    @property
    def name(self):
        return "SwiftNet-PyTorch"

    def load_model(self, model_path, image_size=(1024, 2048)):
        """
        image_size: (H, W) — expected input size
        """
        self.image_size = image_size
        self.target_size = (image_size[0] // 4, image_size[1] // 4)

        resnet = resnet18(pretrained=True, efficient=False)
        model = SemsegModel(resnet, NUM_CLASSES, k=3, bias=False)
        model.upsample_logits = False
        model.criterion = SemsegCrossEntropy(num_classes=NUM_CLASSES, ignore_id=NUM_CLASSES)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval().to(self.device)
        self.model = model

        return 0.0  # no build time for PyTorch

    def prepare_input(self, images):
        return images.to(self.device)

    def run(self, input_data):
        with torch.no_grad():
            logits, _ = self.model(input_data, self.target_size, self.image_size)
        return logits

    def get_logits_from_output(self, output):
        return [output[i].detach().cpu().numpy() for i in range(output.shape[0])]

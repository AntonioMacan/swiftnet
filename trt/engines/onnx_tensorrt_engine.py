import numpy as np
import onnx
import time
import onnx_tensorrt.backend as backend
from .base_engine import InferenceEngine

class ONNXTensorRTEngine(InferenceEngine):
    def __init__(self):
        self.engine = None

    @property
    def name(self):
        return "SwiftNet-TensorRT-ONNX"

    def load_model(self, model_path):
        onnx_model = onnx.load(model_path)
        start_time = time.time()
        self.engine = backend.prepare(onnx_model, device="CUDA:0")
        return time.time() - start_time

    def prepare_input(self, images):
        return images.numpy().astype(np.float32)

    def run(self, input_data):
        return self.engine.run(input_data)

    def get_logits_from_output(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        return [output[i] for i in range(output.shape[0])]
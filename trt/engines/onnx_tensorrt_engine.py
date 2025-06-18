import numpy as np
import torch
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
        """Prepare input for ONNX-TensorRT engine"""
        if isinstance(images, torch.Tensor):
            return images.detach().cpu().numpy().astype(np.float32)
        return images.astype(np.float32)

    def run(self, input_data):
        return self.engine.run(input_data)

    def get_logits_from_output(self, output):
        """Convert output to list of per-image logits tensors for evaluation"""
        if isinstance(output, (tuple, list)):
            output = output[0]
        
        # Convert to torch tensor for consistency with evaluation code
        if isinstance(output, np.ndarray):
            output_tensor = torch.from_numpy(output.copy())
        else:
            output_tensor = output
            
        # Return list of tensors, one per batch item
        return [output_tensor[i] for i in range(output_tensor.shape[0])]
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from .base_engine import InferenceEngine

class TensorRTEngine(InferenceEngine):
    def __init__(self):
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()
        self.d_input = None
        self.d_output = None
        self.h_output = None
        self.output_shape = (1, 19, 1024, 2048)

    @property
    def name(self):
        return "SwiftNet-TensorRT"

    def load_model(self, model_path, workspace_size=1 << 30):
        start_time = time.time()

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError("Failed to parse ONNX model.")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        serialized_engine = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * np.float32().nbytes))
        self.h_output = cuda.pagelocked_empty(shape=self.output_shape, dtype=np.float32)

        return time.time() - start_time

    def prepare_input(self, images):
        arr = images.numpy().astype(np.float32)
        self.d_input = cuda.mem_alloc(arr.nbytes)
        return arr

    def run(self, input_data):
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output

    def get_logits_from_output(self, output):
        return [output[i] for i in range(output.shape[0])]
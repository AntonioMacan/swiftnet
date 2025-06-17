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
        self.output_shape = None

    @property
    def name(self):
        return "SwiftNet-TensorRT"

    def load_model(self, model_path, input_shape, output_channels=19, workspace_size=1 << 30):
        """
        Builds TensorRT engine from ONNX.

        input_shape: (B, C, H, W) — dummy input shape
        """
        B, C, H, W = input_shape
        output_shape = (B, output_channels, H, W)
        self.output_shape = output_shape

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

        # Allocate memory
        input_nbytes = B * C * H * W * np.float32().nbytes
        output_nbytes = np.prod(output_shape) * np.float32().nbytes
        self.d_input = cuda.mem_alloc(int(input_nbytes))
        self.d_output = cuda.mem_alloc(int(output_nbytes))
        self.h_output = cuda.pagelocked_empty(shape=output_shape, dtype=np.float32)

        return time.time() - runtime.start_time if hasattr(runtime, "start_time") else 0.0

    def prepare_input(self, images):
        return images.numpy().astype(np.float32)

    def run(self, input_data):
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output

    def get_logits_from_output(self, output):
        return [output[i] for i in range(output.shape[0])]

    # === Caching ===
    def save_engine(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plan = self.engine.serialize()
        with open(path, "wb") as f:
            f.write(plan)
        print(f"[INFO] Engine cached at {path}")

    @classmethod
    def load_engine(cls, path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to load engine from {path}")

        instance = cls()
        instance.engine = engine
        instance.context = engine.create_execution_context()
        return instance

    @classmethod
    def load_or_build(cls, model_path, cache_path, input_shape):
        if os.path.exists(cache_path):
            try:
                instance = cls.load_engine(cache_path)
                print(f"[INFO] Loaded engine from cache: {cache_path}")
                return instance, 0.0
            except Exception as e:
                print(f"[WARNING] Cache load failed: {e}, rebuilding...")

        instance = cls()
        build_time = instance.load_model(model_path, input_shape)
        try:
            instance.save_engine(cache_path)
        except Exception as e:
            print(f"[WARNING] Failed to save engine cache: {e}")
        return instance, build_time

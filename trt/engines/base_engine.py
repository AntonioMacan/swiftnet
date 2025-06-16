from abc import ABC, abstractmethod

class InferenceEngine(ABC):
    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def prepare_input(self, images):
        pass

    @abstractmethod
    def run(self, input_data):
        pass

    @abstractmethod
    def get_logits_from_output(self, output):
        pass

    @property
    @abstractmethod
    def name(self):
        pass
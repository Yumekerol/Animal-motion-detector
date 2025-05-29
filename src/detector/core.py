import os
from pathlib import Path
from .dataset_handler import DatasetHandler
from .training import ModelTrainer
from .testing import ModelTester
from .evaluation import ModelEvaluator

class RoboflowAnimalDetector:
    def __init__(self):
        self.dataset_handler = DatasetHandler()
        self.trainer = ModelTrainer(self.dataset_handler)
        self.tester = ModelTester(self.dataset_handler)
        self.evaluator = ModelEvaluator(self.dataset_handler)
        self.setup_folders()

    def setup_folders(self):
        self.dataset_handler.setup_folders()

    def download_and_merge_datasets(self):
        return self.dataset_handler.download_and_merge_datasets()

    def load_dataset_config(self):
        return self.dataset_handler.load_dataset_config()

    def validate_dataset(self):
        return self.dataset_handler.validate_dataset()

    def train_model(self, epochs=100, model_size='l'):
        return self.trainer.train_model(epochs, model_size)

    def test_model_webcam(self, model_path="models/best11.pt"):
        self.tester.test_model_webcam(model_path)

    def test_model_on_image(self, model_path="models/best11.pt"):
        self.tester.test_model_on_image(model_path)

    def test_model_on_video(self, model_path="models/best11.pt"):
        self.tester.test_model_on_video(model_path)

    def evaluate_model(self, model_path="models/best_merged.pt"):
        self.evaluator.evaluate_model(model_path)

    @property
    def merged_dataset_path(self):
        return self.dataset_handler.merged_dataset_path

    @property
    def class_names(self):
        return self.dataset_handler.class_names

    @property
    def num_classes(self):
        return self.dataset_handler.num_classes
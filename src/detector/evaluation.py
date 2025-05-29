import os
from ultralytics import YOLO

class ModelEvaluator:
    def __init__(self, dataset_handler):
        self.dataset_handler = dataset_handler

    def evaluate_model(self, model_path="models/best_merged.pt"):
        if not os.path.exists(model_path):
            print("❌ Model not found!")
            return

        if not os.path.exists(self.dataset_handler.merged_dataset_path):
            print("❌ Merged dataset not available!")
            return

        print("📊 Evaluating model performance...")

        try:
            model = YOLO(model_path)
            data_yaml = os.path.join(self.dataset_handler.merged_dataset_path, "data.yaml")
            results = model.val(data=data_yaml, split='test')

            print("✅ Evaluation completed!")
            print(f"📈 Results saved in: runs/detect/val/")

        except Exception as e:
            print(f"❌ Evaluation error: {e}")
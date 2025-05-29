import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

class ModelTrainer:
    def __init__(self, dataset_handler):
        self.dataset_handler = dataset_handler

    def train_model(self, epochs=100, model_size='l'):
        if not os.path.exists(self.dataset_handler.merged_dataset_path):
            print("❌ Merged dataset not available! Download and merge datasets first.")
            return False

        print(f"🚀 Starting training with {epochs} epochs...")
        print(f"📊 Training on {self.dataset_handler.num_classes} classes: {', '.join(self.dataset_handler.class_names)}")

        try:
            model_name = f'yolov8{model_size}.pt'
            model = YOLO(model_name)

            data_yaml = os.path.join(self.dataset_handler.merged_dataset_path, "data.yaml")

            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=640,
                batch=16,
                name='merged_animal_detector',
                patience=20,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                workers=4,
                cache=False,
                verbose=True,
                save_period=10,
                val=True,
                plots=True
            )

            print("✅ Training completed!")

            run_dir = self.find_latest_run_dir()
            if run_dir:
                best_path = os.path.join(run_dir, "weights", "best.pt")
                if os.path.exists(best_path):
                    shutil.copy(best_path, "models/best_merged.pt")
                    print("📋 Model saved to: models/best_merged.pt")

                    last_path = os.path.join(run_dir, "weights", "last.pt")
                    if os.path.exists(last_path):
                        shutil.copy(last_path, "models/last_merged.pt")
                        print("📋 Last model saved to: models/last_merged.pt")

            return True

        except Exception as e:
            print(f"❌ Training error: {e}")
            return False

    def find_latest_run_dir(self):
        runs_dir = Path("runs/detect")
        if not runs_dir.exists():
            return None

        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "merged_animal_detector" in d.name]
        if not run_dirs:
            return None

        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        return str(latest_run)
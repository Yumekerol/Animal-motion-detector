from ultralytics import YOLO
import os
from pathlib import Path


class AnimalTrainer:
    def __init__(self):
        self.data_path = Path("data/dataset.yaml")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def train_model(self, epochs=50, batch_size=16, img_size=640):
        """Train custom YOLO model"""

        # Check if dataset exists
        if not self.data_path.exists():
            print("❌ Dataset not found!")
            print("Run first: python src/collect_data.py")
            return

        print("🚀 Starting model training...")
        print(f"📊 Configuration:")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Image size: {img_size}")

        # Load base model
        model = YOLO('yolov8n.pt')  # nano version for faster training

        try:
            # Train model
            results = model.train(
                data=str(self.data_path),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                name='animal_detector',
                save=True,
                plots=True,
                device='cpu'  # or 'cuda' if you have GPU
            )

            print("✅ Training completed!")
            print(f"📁 Model saved at: runs/detect/animal_detector/weights/best.pt")

            # Copy best model to models folder
            best_model_path = Path("runs/detect/animal_detector/weights/best.pt")
            if best_model_path.exists():
                import shutil
                shutil.copy(best_model_path, self.models_dir / "best.pt")
                print(f"📋 Model copied to: {self.models_dir / 'best.pt'}")

            return results

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return None

    def evaluate_model(self):
        """Evaluate trained model"""
        model_path = self.models_dir / "best.pt"

        if not model_path.exists():
            print("❌ Trained model not found!")
            print("Run first: python src/train_model.py")
            return

        print("📊 Evaluating model...")

        # Load trained model
        model = YOLO(str(model_path))

        # Evaluate on validation dataset
        results = model.val(data=str(self.data_path))

        print("✅ Evaluation completed!")
        print(f"📈 Metrics saved at: runs/detect/val/")

        return results

    def test_model_webcam(self):
        """Test trained model with webcam"""
        model_path = self.models_dir / "best.pt"

        if not model_path.exists():
            print("❌ Trained model not found!")
            return

        print("🎥 Testing model with webcam...")

        # Load model
        model = YOLO(str(model_path))

        # Test with webcam
        results = model.predict(
            source=0,  # webcam
            show=True,
            conf=0.5,
            save=True,
            name='webcam_test'
        )

        print("✅ Webcam test completed!")


def main():
    trainer = AnimalTrainer()

    print("🐾 Animal Detection Model Trainer")
    print("Choose an option:")
    print("1. Train model (full training)")
    print("2. Quick training (10 epochs)")
    print("3. Evaluate trained model")
    print("4. Test model with webcam")

    choice = input("Enter your choice (1-4): ").strip()

    if choice == "1":
        print("🚀 Starting full training...")
        trainer.train_model(epochs=50)

    elif choice == "2":
        print("⚡ Starting quick training...")
        trainer.train_model(epochs=10, batch_size=8)

    elif choice == "3":
        trainer.evaluate_model()

    elif choice == "4":
        trainer.test_model_webcam()

    else:
        print("❌ Invalid option")


if __name__ == "__main__":
    main()
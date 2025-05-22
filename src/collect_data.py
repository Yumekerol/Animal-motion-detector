import os
import requests
import zipfile
from ultralytics import YOLO
import cv2
import json
from pathlib import Path


class DatasetCollector:
    def __init__(self):
        self.data_dir = Path("data")
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"

        # Create folder structure
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

    def download_sample_dataset(self):
        """Download sample dataset from Roboflow"""
        print("🔄 Downloading sample dataset...")

        # URL of a public Roboflow dataset (example)
        # You can replace with any public dataset
        url = "https://public.roboflow.com/ds/YOUR_DATASET_URL"

        # For now, let's create some sample data
        self.create_sample_data()

    def create_sample_data(self):
        """Create sample data for testing"""
        print("📝 Creating sample data...")

        # Use pre-trained model to create automatic annotations
        model = YOLO('yolov8n.pt')

        # Capture some images from webcam
        self.capture_webcam_images(model)

    def capture_webcam_images(self, model, num_images=20):
        """Capture images from webcam and create automatic annotations"""
        print(f"📸 Capturing {num_images} images from webcam...")
        print("Position animals in front of the camera and press SPACE to capture")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(0)
        captured = 0

        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                break

            # Show current frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Captured: {captured}/{num_images}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE=capture, Q=quit",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Dataset Capture', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                # Detect objects in image
                results = model(frame, classes=[15, 16])  # cat, dog

                # If animals found, save
                if self.has_animals(results):
                    self.save_image_with_labels(frame, results, captured)
                    captured += 1
                    print(f"✅ Image {captured} captured and annotated!")
                else:
                    print("❌ No animals detected in this image")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Dataset created with {captured} images!")

    def has_animals(self, results):
        """Check if detection found animals"""
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                return True
        return False

    def save_image_with_labels(self, frame, results, img_id):
        """Save image and its annotations"""
        # Split between train/val (80/20)
        split = 'train' if img_id < 16 else 'val'

        # Save image
        img_filename = f"animal_{img_id:03d}.jpg"
        img_path = self.images_dir / split / img_filename
        cv2.imwrite(str(img_path), frame)

        # Create YOLO annotations
        height, width = frame.shape[:2]
        label_path = self.labels_dir / split / f"animal_{img_id:03d}.txt"

        with open(label_path, 'w') as f:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Convert to YOLO format
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])

                        # Normalize coordinates
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height

                        # Convert class (15=cat->0, 16=dog->1)
                        yolo_class = 0 if cls == 15 else 1

                        f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    def create_dataset_yaml(self):
        """Create dataset configuration file"""
        yaml_content = f"""path: {self.data_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['cat', 'dog']
"""

        yaml_path = self.data_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"✅ Dataset YAML created at: {yaml_path}")

    def download_online_dataset(self):
        """Alternative: download online dataset"""
        print("🌐 Online dataset options:")
        print("1. Roboflow Universe - https://universe.roboflow.com/")
        print("2. Kaggle Dogs vs Cats - https://www.kaggle.com/c/dogs-vs-cats")
        print("3. Open Images Dataset")

        print("\n💡 To use Roboflow dataset:")
        print("1. Go to https://universe.roboflow.com/")
        print("2. Search for 'animal detection' or 'pet detection'")
        print("3. Choose a public dataset")
        print("4. Download in YOLOv8 format")
        print("5. Extract to 'data/' folder")


def main():
    collector = DatasetCollector()

    print("🐾 Dataset Collector for Animal Detection")
    print("Choose an option:")
    print("1. Capture images from webcam (recommended to start)")
    print("2. Download online dataset")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        collector.create_sample_data()
        collector.create_dataset_yaml()
        print("\n🎉 Basic dataset created!")
        print("Next step: python src/train_model.py")

    elif choice == "2":
        collector.download_online_dataset()

    else:
        print("❌ Invalid option")


if __name__ == "__main__":
    main()
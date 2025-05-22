from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import shutil
import numpy as np


class FixedAnimalDetector:
    def __init__(self):
        self.setup_folders()
        self.create_dataset_yaml()

    def setup_folders(self):
        """Create folder structure"""
        folders = [
            "data/images/train",
            "data/images/val",
            "data/images/test",
            "data/labels/train",
            "data/labels/val",
            "data/labels/test",
            "models",
            "outputs"
        ]

        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

        print("✅ Folder structure created!")

    def create_dataset_yaml(self):
        """Create dataset.yaml file with absolute path"""
        data_path = Path("data").resolve()
        yaml_content = f"""path: {data_path}
train: images/train
val: images/val
test: images/test

nc: 3
names: ['cat', 'dog', 'person']
"""

        with open("data/dataset.yaml", "w") as f:
            f.write(yaml_content)

        print("✅ Dataset YAML created!")

    def fix_existing_labels(self):
        """Fix corrupted label files"""
        print("🔧 Fixing existing label files...")

        label_dirs = ["data/labels/train", "data/labels/val"]
        fixed_count = 0

        for label_dir in label_dirs:
            label_path = Path(label_dir)
            if not label_path.exists():
                continue

            for label_file in label_path.glob("*.txt"):
                try:
                    # Read the corrupted file
                    with open(label_file, 'r') as f:
                        content = f.read()

                    # Fix the newline characters
                    fixed_content = content.replace('\\n', '\n')

                    # Remove any trailing characters that aren't numbers or spaces
                    lines = []
                    for line in fixed_content.split('\n'):
                        line = line.strip()
                        if line and not line.endswith(('\\', 'n')):
                            # Validate YOLO format: class x_center y_center width height
                            parts = line.split()
                            if len(parts) == 5:
                                try:
                                    cls = int(parts[0])
                                    coords = [float(x) for x in parts[1:]]
                                    # Ensure coordinates are in valid range [0,1]
                                    if all(0 <= coord <= 1 for coord in coords) and cls in [0, 1, 2]:
                                        lines.append(line)
                                except ValueError:
                                    continue

                    # Write the fixed content
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(lines))
                        if lines:  # Add final newline if there's content
                            f.write('\n')

                    fixed_count += 1
                    print(f"✅ Fixed {label_file.name}")

                except Exception as e:
                    print(f"❌ Error fixing {label_file.name}: {e}")

        print(f"🔧 Fixed {fixed_count} label files!")

        # Clean cache files to force regeneration
        cache_files = ["data/labels/train.cache", "data/labels/val.cache"]
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️ Removed {cache_file}")

    def collect_images(self, num_images=15):
        """Collect images from webcam with proper label formatting"""
        print(f"📸 Let's capture {num_images} images!")
        print("Instructions:")
        print("- Place animals or people in front of the camera")
        print("- Press SPACE to capture")
        print("- Press Q to quit")

        model = YOLO('yolov8n.pt')
        cap = cv2.VideoCapture(0)
        captured = 0

        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                print("❌ Webcam error")
                break

            display_frame = frame.copy()
            cv2.putText(display_frame, f"Captures: {captured}/{num_images}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE=capture, Q=quit",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Dataset Capture', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Detect animals and people (cat=15, dog=16, person=0 in COCO)
                results = model(frame, classes=[0, 15, 16], verbose=False)

                if self.has_targets(results):
                    self.save_image_and_label(frame, results, captured)
                    captured += 1
                    print(f"✅ Image {captured} captured!")
                else:
                    print("❌ No animals or people detected")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Dataset created with {captured} images!")

    def has_targets(self, results):
        """Check if animals or people were detected"""
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                return True
        return False

    def save_image_and_label(self, frame, results, img_id):
        """Save image and label with proper formatting"""
        # Split data: 80% train, 20% val
        split = 'train' if img_id < int(0.8 * 15) else 'val'

        # Save image
        img_filename = f"animal_{img_id:03d}.jpg"
        img_path = f"data/images/{split}/{img_filename}"
        cv2.imwrite(img_path, frame)

        # Save labels in proper YOLO format
        height, width = frame.shape[:2]
        label_path = f"data/labels/{split}/animal_{img_id:03d}.txt"

        with open(label_path, 'w') as f:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])

                        # Convert to YOLO format (normalized)
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height

                        # Ensure values are in valid range
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        w = max(0, min(1, w))
                        h = max(0, min(1, h))

                        # Convert COCO class to YOLO class
                        # COCO: person=0, cat=15, dog=16
                        # YOLO: cat=0, dog=1, person=2
                        if cls == 15:  # cat
                            yolo_class = 0
                        elif cls == 16:  # dog
                            yolo_class = 1
                        elif cls == 0:   # person
                            yolo_class = 2
                        else:
                            continue  # Skip unknown classes

                        # Write with proper formatting
                        f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    def validate_dataset(self):
        """Validate dataset before training"""
        print("🔍 Validating dataset...")

        train_images = list(Path("data/images/train").glob("*.jpg"))
        train_labels = list(Path("data/labels/train").glob("*.txt"))
        val_images = list(Path("data/images/val").glob("*.jpg"))
        val_labels = list(Path("data/labels/val").glob("*.txt"))

        print(f"📊 Training: {len(train_images)} images, {len(train_labels)} labels")
        print(f"📊 Validation: {len(val_images)} images, {len(val_labels)} labels")

        # Check for label issues
        issues = 0
        for label_file in train_labels + val_labels:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        print(f"❌ {label_file.name} line {i + 1}: Expected 5 values, got {len(parts)}")
                        issues += 1
                        continue

                    try:
                        cls = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        if cls not in [0, 1, 2]:  # cat, dog, person
                            print(f"❌ {label_file.name} line {i + 1}: Invalid class {cls}")
                            issues += 1

                        if not all(0 <= coord <= 1 for coord in coords):
                            print(f"❌ {label_file.name} line {i + 1}: Coordinates out of range")
                            issues += 1

                    except ValueError as e:
                        print(f"❌ {label_file.name} line {i + 1}: {e}")
                        issues += 1

            except Exception as e:
                print(f"❌ Error reading {label_file.name}: {e}")
                issues += 1

        if issues == 0:
            print("✅ Dataset validation passed!")
            return True
        else:
            print(f"❌ Found {issues} issues in dataset")
            return False

    def train_model(self, epochs=50):
        """Train model with better error handling"""
        print(f"🚀 Starting training with {epochs} epochs...")

        # Validate dataset first
        if not self.validate_dataset():
            print("❌ Dataset validation failed! Fix issues before training.")
            return False

        try:
            # Load model
            model = YOLO('yolov8n.pt')

            # Train with better parameters
            results = model.train(
                data='data/dataset.yaml',
                epochs=epochs,
                imgsz=640,
                batch=2,  # Smaller batch size for stability
                name='animal_detector',
                patience=15,
                device='cpu',
                workers=1,  # Reduce workers for stability
                cache=False,  # Disable caching to avoid issues
                verbose=True
            )

            print("✅ Training completed!")

            # Copy best model
            best_path = "runs/detect/animal_detector/weights/best.pt"
            if os.path.exists(best_path):
                shutil.copy(best_path, "models/best.pt")
                print("📋 Model saved to: models/best.pt")
            else:
                # Try alternative path
                alt_path = "runs/detect/animal_detector2/weights/best.pt"
                if os.path.exists(alt_path):
                    shutil.copy(alt_path, "models/best.pt")
                    print("📋 Model saved to: models/best.pt")

            return True

        except Exception as e:
            print(f"❌ Training error: {e}")
            print("\n🔧 Troubleshooting tips:")
            print("1. Check if all label files are properly formatted")
            print("2. Ensure images and labels have matching names")
            print("3. Verify dataset.yaml path is correct")
            return False

    def test_model(self):
        """Test trained model"""
        model_path = "models/best.pt"

        if not os.path.exists(model_path):
            print("❌ Trained model not found!")
            print("Run training first")
            return

        print("🎥 Testing custom model...")
        print("Press Q to quit")

        custom_model = YOLO(model_path)
        original_model = YOLO('yolov8n.pt')

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get predictions
            custom_results = custom_model(frame, conf=0.5, verbose=False)
            original_results = original_model(frame, classes=[0, 15, 16], conf=0.5, verbose=False)

            # Create side-by-side comparison
            height, width = frame.shape[:2]
            display_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)

            # Custom model results (left side)
            left_frame = frame.copy()
            self.draw_detections(left_frame, custom_results, "CUSTOM MODEL", (0, 255, 0))
            display_frame[0:height, 0:width] = left_frame

            # Original model results (right side)
            right_frame = frame.copy()
            self.draw_detections(right_frame, original_results, "ORIGINAL MODEL", (255, 0, 0))
            display_frame[0:height, width:width * 2] = right_frame

            cv2.imshow('Model Comparison', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_detections(self, frame, results, title, color):
        """Draw detections on frame"""
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        count = 0
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if hasattr(result, 'names'):
                        class_name = result.names[cls]
                    else:
                        # For custom model: 0=cat, 1=dog, 2=person
                        class_names = ['cat', 'dog', 'person']
                        class_name = class_names[cls] if cls < 3 else 'unknown'

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{class_name} {conf:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    count += 1

        cv2.putText(frame, f'Detections: {count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    detector = FixedAnimalDetector()

    print("🐾 Animal and Person Detector")
    print("=" * 50)
    print("1. Fix existing corrupted labels")
    print("2. Collect new images (webcam)")
    print("3. Validate dataset")
    print("4. Train model")
    print("5. Test model")
    print("6. Run full pipeline")
    print("=" * 50)

    choice = input("Enter your choice (1-6): ").strip()

    if choice == "1":
        detector.fix_existing_labels()

    elif choice == "2":
        detector.collect_images()

    elif choice == "3":
        detector.validate_dataset()

    elif choice == "4":
        detector.train_model()

    elif choice == "5":
        detector.test_model()

    elif choice == "6":
        print("🚀 Running full pipeline...")

        print("\n🔧 Step 1: Fix existing labels")
        detector.fix_existing_labels()

        print("\n📸 Step 2: Data collection")
        detector.collect_images(20)

        print("\n🔍 Step 3: Validate dataset")
        if detector.validate_dataset():
            print("\n🚀 Step 4: Model training")
            if detector.train_model(30):
                print("\n🎥 Step 5: Model testing")
                detector.test_model()
        else:
            print("❌ Dataset validation failed. Please fix issues first.")

    else:
        print("❌ Invalid option")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install ultralytics opencv-python numpy")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
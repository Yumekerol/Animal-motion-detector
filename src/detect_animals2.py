from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import shutil
import numpy as np
import yaml
from roboflow import Roboflow
import torch
from collections import defaultdict
import random


class RoboflowAnimalDetector:
    def __init__(self):
        self.dataset_paths = []
        self.merged_dataset_path = "merged_dataset"
        self.class_names = []
        self.num_classes = 0
        self.setup_folders()

    def setup_folders(self):
        folders = [
            "models",
            "outputs",
            "runs",
            "datasets",
            self.merged_dataset_path
        ]

        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

        print("✅ Folder structure created!")

    def download_multiple_datasets(self):
        print("📥 Downloading dataset from Roboflow...")

        datasets_config = [
            {
                "api_key": "TVcshbt65yMqspCiLOUN",
                "workspace": "araa-gjjv3",
                "project": "klasifikasi-hewan-ne5og",
                "version": 1,
                "name": "dataset1"
            }
        ]

        self.dataset_paths = []

        for i, config in enumerate(datasets_config):
            try:
                print(f"\n📂 Downloading dataset {i + 1}: {config['name']}")

                rf = Roboflow(api_key=config["api_key"])
                project = rf.workspace(config["workspace"]).project(config["project"])
                version = project.version(config["version"])

                dataset_folder = f"datasets/{config['name']}"
                dataset = version.download("yolov8", location=dataset_folder)

                self.dataset_paths.append(dataset.location)
                print(f"✅ Dataset {i + 1} downloaded to: {dataset.location}")

            except Exception as e:
                print(f"❌ Error downloading dataset {i + 1}: {e}")
                return False

        if len(self.dataset_paths) > 0:
            print(f"\n✅ Successfully downloaded {len(self.dataset_paths)} datasets!")
            return True
        else:
            print("❌ No datasets were downloaded successfully!")
            return False

    def merge_datasets(self):
        if not self.dataset_paths:
            print("❌ No datasets to merge! Download datasets first.")
            return False

        print("🔄 Merging datasets...")

        splits = ['train', 'valid', 'test']
        for split in splits:
            for subfolder in ['images', 'labels']:
                Path(self.merged_dataset_path, split, subfolder).mkdir(parents=True, exist_ok=True)

        all_classes = set()
        dataset_configs = []

        for dataset_path in self.dataset_paths:
            yaml_path = os.path.join(dataset_path, "data.yaml")
            try:
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                    dataset_configs.append(config)
                    all_classes.update(config.get('names', []))
            except Exception as e:
                print(f"❌ Error reading config from {dataset_path}: {e}")
                return False

        self.class_names = sorted(list(all_classes))
        self.num_classes = len(self.class_names)
        class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        print(f"📊 Unified dataset will have {self.num_classes} classes:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")

        file_counter = 0

        for dataset_idx, (dataset_path, config) in enumerate(zip(self.dataset_paths, dataset_configs)):
            print(f"\n🔄 Processing dataset {dataset_idx + 1}...")

            old_names = config.get('names', [])
            class_mapping = {}
            for old_id, old_name in enumerate(old_names):
                if old_name in class_to_id:
                    class_mapping[old_id] = class_to_id[old_name]

            for split in splits:
                source_images = Path(dataset_path, split, 'images')
                source_labels = Path(dataset_path, split, 'labels')

                if not source_images.exists():
                    source_images = Path(dataset_path, split)
                    source_labels = Path(dataset_path, split)

                if source_images.exists():
                    # Get all image files
                    image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png")) + list(
                        source_images.glob("*.jpeg"))

                    for image_file in image_files:
                        # Copy image with new name
                        new_image_name = f"img_{file_counter:06d}{image_file.suffix}"
                        dest_image = Path(self.merged_dataset_path, split, 'images', new_image_name)
                        shutil.copy2(image_file, dest_image)

                        label_file = source_labels / f"{image_file.stem}.txt"
                        if label_file.exists():
                            dest_label = Path(self.merged_dataset_path, split, 'labels', f"img_{file_counter:06d}.txt")

                            # Read original labels and convert class IDs
                            with open(label_file, 'r') as f:
                                lines = f.readlines()

                            with open(dest_label, 'w') as f:
                                for line in lines:
                                    line = line.strip()
                                    if line:
                                        parts = line.split()
                                        if len(parts) >= 5:
                                            old_class_id = int(parts[0])
                                            if old_class_id in class_mapping:
                                                new_class_id = class_mapping[old_class_id]
                                                new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                                                f.write(new_line)

                        file_counter += 1

                    print(f"  📂 {split}: processed {len(image_files)} images")

        merged_config = {
            'path': os.path.abspath(self.merged_dataset_path),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': self.num_classes,
            'names': self.class_names
        }

        with open(os.path.join(self.merged_dataset_path, 'data.yaml'), 'w') as f:
            yaml.dump(merged_config, f, default_flow_style=False)

        print(f"\n✅ Dataset merging completed!")
        print(f"📊 Merged dataset statistics:")

        for split in splits:
            split_images = Path(self.merged_dataset_path, split, 'images')
            split_labels = Path(self.merged_dataset_path, split, 'labels')

            if split_images.exists():
                num_images = len(list(split_images.glob("*.jpg")) + list(split_images.glob("*.png")) + list(
                    split_images.glob("*.jpeg")))
                num_labels = len(list(split_labels.glob("*.txt"))) if split_labels.exists() else 0
                print(f"  📂 {split.capitalize()}: {num_images} images, {num_labels} labels")

        return True

    def download_roboflow_dataset(self):
        return self.download_and_merge_datasets()

    def download_and_merge_datasets(self):
        if self.download_multiple_datasets():
            return self.merge_datasets()
        return False

    def load_dataset_config(self):
        yaml_path = os.path.join(self.merged_dataset_path, "data.yaml")

        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            self.num_classes = config.get('nc', 0)
            self.class_names = config.get('names', [])

            print(f"✅ Merged dataset loaded with {self.num_classes} classes:")
            for i, name in enumerate(self.class_names):
                print(f"  {i}: {name}")

            return True

        except Exception as e:
            print(f"❌ Error loading dataset config: {e}")
            return False

    def validate_dataset(self):
        if not os.path.exists(self.merged_dataset_path):
            print("❌ Merged dataset not found!")
            return False

        print("🔍 Validating merged dataset...")

        required_folders = ['train', 'valid', 'test']
        for folder in required_folders:
            images_path = os.path.join(self.merged_dataset_path, folder, 'images')
            labels_path = os.path.join(self.merged_dataset_path, folder, 'labels')

            if os.path.exists(images_path):
                images = list(Path(images_path).glob("*.jpg")) + list(Path(images_path).glob("*.png")) + list(
                    Path(images_path).glob("*.jpeg"))
                labels = list(Path(labels_path).glob("*.txt")) if os.path.exists(labels_path) else []
                print(f"📊 {folder.capitalize()}: {len(images)} images, {len(labels)} labels")
            else:
                print(f"⚠️ {folder} folder not found")

        # Validate some label files
        train_labels_path = os.path.join(self.merged_dataset_path, "train", "labels")
        if os.path.exists(train_labels_path):
            label_files = list(Path(train_labels_path).glob("*.txt"))
            issues = 0

            for label_file in label_files[:20]:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) != 5:
                            issues += 1
                            continue

                        try:
                            cls = int(parts[0])
                            coords = [float(x) for x in parts[1:]]

                            if cls < 0 or cls >= self.num_classes:
                                issues += 1

                            if not all(0 <= coord <= 1 for coord in coords):
                                issues += 1

                        except ValueError:
                            issues += 1

                except Exception:
                    issues += 1

            if issues == 0:
                print("✅ Dataset validation passed!")
                return True
            else:
                print(f"⚠️ Found {issues} potential issues, but proceeding with training")
                return True

        return False

    def train_model(self, epochs=100, model_size='l'):
        if not os.path.exists(self.merged_dataset_path):
            print("❌ Merged dataset not available! Download and merge datasets first.")
            return False

        print(f"🚀 Starting training with {epochs} epochs...")
        print(f"📊 Training on {self.num_classes} classes: {', '.join(self.class_names)}")

        try:
            model_name = f'yolov8{model_size}.pt'
            model = YOLO(model_name)

            data_yaml = os.path.join(self.merged_dataset_path, "data.yaml")

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

    def test_model_webcam(self, model_path="models/best8.pt"):
        if not os.path.exists(model_path):
            print("❌ Trained model not found!")
            print(f"Expected path: {model_path}")
            return

        print("🎥 Testing merged model with webcam...")
        print("📋 Classes that can be detected:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
        print("\n🎮 Controls:")
        print("  Q - Quit")
        print("  S - Save screenshot")

        model = YOLO(model_path)
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        screenshot_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break

            results = model(frame, conf=0.3, verbose=False)

            annotated_frame = self.draw_detections(frame, results)

            cv2.imshow('Merged Animal Detector - Webcam', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"outputs/webcam_screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
                screenshot_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def test_model_on_image(self, model_path="models/best8.pt"):
        if not os.path.exists(model_path):
            print("❌ Trained model not found!")
            print(f"Expected path: {model_path}")
            return

        image_path = input("Enter the path to your image file: ").strip().strip('"\'')

        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return

        print(f"🔍 Testing model on image: {image_path}")
        print("📋 Classes that can be detected:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")

        try:
            model = YOLO(model_path)

            image = cv2.imread(image_path)
            if image is None:
                print("❌ Could not load image. Please check the file format.")
                return

            results = model(image, conf=0.3, verbose=False)
            annotated_image = self.draw_detections(image, results)

            output_filename = f"detected_{Path(image_path).stem}{Path(image_path).suffix}"
            output_path = f"outputs/{output_filename}"
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Result saved to: {output_path}")

            print("\n🎮 Controls:")
            print("  Any key - Close window")

            height, width = annotated_image.shape[:2]
            if height > 800 or width > 1200:
                scale = min(800 / height, 1200 / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                annotated_image = cv2.resize(annotated_image, (new_width, new_height))

            cv2.imshow('Animal Detection Result - Image', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"❌ Error processing image: {e}")

    def test_model_on_video(self, model_path="models/best8.pt"):
        """Test model on video file"""
        if not os.path.exists(model_path):
            print("❌ Trained model not found!")
            print(f"Expected path: {model_path}")
            return

        video_path = input("Enter the path to your video file: ").strip().strip('"\'')

        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return

        print(f"🎬 Testing model on video: {video_path}")
        print("📋 Classes that can be detected:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
        print("\n🎮 Controls:")
        print("  Q - Quit")
        print("  SPACE - Pause/Resume")
        print("  S - Save current frame")

        try:
            model = YOLO(model_path)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("❌ Could not open video file")
                return

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            print(f"📊 Video info: {total_frames} frames, {fps} FPS, {duration:.1f}s duration")

            save_video = input("Save processed video? (y/n): ").strip().lower() == 'y'

            video_writer = None
            if save_video:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_video_path = f"outputs/detected_{Path(video_path).stem}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                print(f"📹 Output video will be saved to: {output_video_path}")

            frame_count = 0
            screenshot_count = 0
            paused = False

            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("✅ Video processing completed!")
                        break

                    frame_count += 1
                    results = model(frame, conf=0.3, verbose=False)
                    annotated_frame = self.draw_detections(frame, results)

                    cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                                (10, annotated_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if video_writer is not None:
                        video_writer.write(annotated_frame)

                    if frame_count % 30 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"🎬 Processing: {progress:.1f}% ({frame_count}/{total_frames})")

                else:
                    annotated_frame = current_frame

                cv2.imshow('Animal Detection Result - Video', annotated_frame)
                current_frame = annotated_frame

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    status = "PAUSED" if paused else "PLAYING"
                    print(f"🎬 Video {status}")
                elif key == ord('s'):
                    screenshot_path = f"outputs/video_frame_{screenshot_count:03d}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"📸 Frame saved: {screenshot_path}")
                    screenshot_count += 1

            cap.release()
            if video_writer is not None:
                video_writer.release()
                print(f"✅ Output video saved!")
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"❌ Error processing video: {e}")

    def draw_detections(self, frame, results):
        annotated_frame = frame.copy()

        cv2.putText(annotated_frame, "Merged Animal Detector",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        detection_count = 0
        class_counts = {}

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if cls < len(self.class_names):
                        class_name = self.class_names[cls]
                    else:
                        class_name = f'Class_{cls}'

                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
                    color = colors[cls % len(colors)]

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    label = f'{class_name} {conf:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                  (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    detection_count += 1

        y_offset = 70
        cv2.putText(annotated_frame, f'Total Detections: {detection_count}',
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for class_name, count in class_counts.items():
            y_offset += 25
            cv2.putText(annotated_frame, f'{class_name}: {count}',
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    def evaluate_model(self, model_path="models/best_merged.pt"):
        if not os.path.exists(model_path):
            print("❌ Model not found!")
            return

        if not os.path.exists(self.merged_dataset_path):
            print("❌ Merged dataset not available!")
            return

        print("📊 Evaluating model performance...")

        try:
            model = YOLO(model_path)
            data_yaml = os.path.join(self.merged_dataset_path, "data.yaml")
            results = model.val(data=data_yaml, split='test')

            print("✅ Evaluation completed!")
            print(f"📈 Results saved in: runs/detect/val/")

        except Exception as e:
            print(f"❌ Evaluation error: {e}")

def main():
    detector = RoboflowAnimalDetector()

    print("🐾 Enhanced Animal Detector with Multiple Datasets")
    print("=" * 60)
    print("1. Download and merge multiple datasets")
    print("2. Validate merged dataset")
    print("3. Train model (Nano - Fast)")
    print("4. Train model (Small - Balanced)")
    print("5. Train model (Medium - Accurate)")
    print("6. Train model (Large - Very Accurate)")
    print("7. Test model (Webcam)")
    print("8. Test model (Image file)")
    print("9. Test model (Video file)")
    print("10. Evaluate model performance")
    print("11. Run full pipeline")
    print("=" * 60)

    choice = input("Enter your choice (1-11): ").strip()

    if choice == "1":
        detector.download_and_merge_datasets()

    elif choice == "2":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.validate_dataset()

    elif choice == "3":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.train_model(epochs=100, model_size='n')

    elif choice == "4":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.train_model(epochs=200, model_size='s')

    elif choice == "5":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.train_model(epochs=300, model_size='m')

    elif choice == "6":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.train_model(epochs=300, model_size='l')

    elif choice == "7":
        if detector.class_names or (detector.load_dataset_config()):
            detector.test_model_webcam()

    elif choice == "8":
        if detector.class_names or (detector.load_dataset_config()):
            detector.test_model_on_image()

    elif choice == "9":
        if detector.class_names or (detector.load_dataset_config()):
            detector.test_model_on_video()

    elif choice == "10":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.evaluate_model()

    elif choice == "11":
        print("🚀 Running full pipeline...")

        print("\n📥 Step 1: Download and merge datasets")
        if detector.download_and_merge_datasets():
            detector.load_dataset_config()

            print("\n🔍 Step 2: Validate merged dataset")
            if detector.validate_dataset():

                print("\n🚀 Step 3: Train model")
                if detector.train_model(epochs=100, model_size='s'):
                    print("\n📊 Step 4: Evaluate model")
                    detector.evaluate_model()

                    print("\n🎥 Step 5: Test model")
                    test_choice = input("Test with (w)ebcam, (i)mage, or (v)ideo? ").strip().lower()
                    if test_choice == 'w':
                        detector.test_model_webcam()
                    elif test_choice == 'i':
                        detector.test_model_on_image()
                    elif test_choice == 'v':
                        detector.test_model_on_video()

    else:
        print("❌ Invalid option")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install ultralytics opencv-python roboflow pyyaml")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
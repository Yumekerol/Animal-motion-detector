import os
from pathlib import Path
import shutil
import yaml
from roboflow import Roboflow
from collections import defaultdict
import random

class DatasetHandler:
    def __init__(self):
        self.dataset_paths = []
        self.merged_dataset_path = "merged_dataset"
        self.class_names = []
        self.num_classes = 0

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
                "workspace": "bottle-7s3lj",
                "project": "bottle-gcqnx",
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
                    image_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.png")) + list(
                        source_images.glob("*.jpeg"))

                    for image_file in image_files:
                        new_image_name = f"img_{file_counter:06d}{image_file.suffix}"
                        dest_image = Path(self.merged_dataset_path, split, 'images', new_image_name)
                        shutil.copy2(image_file, dest_image)

                        label_file = source_labels / f"{image_file.stem}.txt"
                        if label_file.exists():
                            dest_label = Path(self.merged_dataset_path, split, 'labels', f"img_{file_counter:06d}.txt")

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
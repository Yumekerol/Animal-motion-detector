import os
import cv2
import numpy as np
from pathlib import Path
import yaml
import shutil


def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml_config(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None


def save_yaml_config(file_path, data):
    try:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving YAML file {file_path}: {e}")
        return False


def copy_file(src, dst):
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Error copying file from {src} to {dst}: {e}")
        return False


def resize_image(image, max_height=800, max_width=1200):
    height, width = image.shape[:2]

    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image
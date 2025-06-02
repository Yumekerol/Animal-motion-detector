from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json


class ModelEvaluator:
    def __init__(self, model_path="models/best.pt"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def evaluate_on_test_set(self):
        """Evaluate model on test dataset and get mAP metrics"""
        if not self.model:
            print("❌ Model not loaded!")
            return None

        print("📊 Evaluating model on test dataset...")

        try:
            results = self.model.val(
                data='data/dataset.yaml',
                split='test',
                imgsz=640,
                conf=0.25,
                iou=0.5,
                verbose=True
            )

            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (
                                                                                                                     results.box.mp + results.box.mr) > 0 else 0
            }
            print("\n📈 Evaluation Results:")
            print("=" * 40)
            print(f"mAP@0.5:     {metrics['mAP50']:.3f}")
            print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.3f}")
            print(f"Precision:   {metrics['precision']:.3f}")
            print(f"Recall:      {metrics['recall']:.3f}")
            print(f"F1-Score:    {metrics['f1_score']:.3f}")
            print("=" * 40)

            with open('outputs/evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

            return metrics

        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return None

    def create_confusion_matrix(self):
        """Create and save confusion matrix"""
        if not self.model:
            print("❌ Model not loaded!")
            return

        print("📊 Creating confusion matrix...")

        try:
            results = self.model.val(
                data='data/dataset.yaml',
                split='test',
                save_json=True,
                plots=True
            )

            print("✅ Confusion matrix saved in runs/detect/val/")

        except Exception as e:
            print(f"❌ Error creating confusion matrix: {e}")

    def test_inference_speed(self, num_frames=100):
        """Test inference speed on webcam"""
        if not self.model:
            print("❌ Model not loaded!")
            return

        print(f"⏱️ Testing inference speed on {num_frames} frames...")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return

        times = []

        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            start_time = cv2.getTickCount()
            results = self.model(frame, verbose=False)
            end_time = cv2.getTickCount()

            inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
            times.append(inference_time)

            if i % 20 == 0:
                print(f"Processed {i}/{num_frames} frames...")

        cap.release()

        avg_time = np.mean(times)
        fps = 1000 / avg_time
        min_time = np.min(times)
        max_time = np.max(times)

        print("\n⚡ Performance Results:")
        print("=" * 30)
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"FPS: {fps:.2f}")
        print(f"Min time: {min_time:.2f} ms")
        print(f"Max time: {max_time:.2f} ms")
        print("=" * 30)

        # Save results
        performance_data = {
            'avg_inference_time_ms': avg_time,
            'fps': fps,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'total_frames': len(times)
        }

        with open('outputs/performance_metrics.json', 'w') as f:
            json.dump(performance_data, f, indent=2)

        return performance_data

    def compare_with_baseline(self):
        """Compare custom model with original YOLOv8"""
        if not self.model:
            print("❌ Model not loaded!")
            return

        print("🔍 Comparing with baseline YOLOv8...")

        baseline_model = YOLO('../animal-detector-backend/yolov8n.pt')

        cap = cv2.VideoCapture(0)

        custom_detections = []
        baseline_detections = []

        print("📹 Recording 30 seconds of detections...")
        print("Make sure animals/people are visible!")

        start_time = cv2.getTickCount()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cv2.getTickCount()
            elapsed = (current_time - start_time) / cv2.getTickFrequency()

            if elapsed > 30:  # 30 seconds
                break

            custom_results = self.model(frame, verbose=False)
            baseline_results = baseline_model(frame, classes=[0, 15, 16], verbose=False)  # person, cat, dog

            custom_count = sum([len(r.boxes) if r.boxes is not None else 0 for r in custom_results])
            baseline_count = sum([len(r.boxes) if r.boxes is not None else 0 for r in baseline_results])

            custom_detections.append(custom_count)
            baseline_detections.append(baseline_count)

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames ({elapsed:.1f}s)")

        cap.release()

        custom_avg = np.mean(custom_detections)
        baseline_avg = np.mean(baseline_detections)

        print("\n📊 Comparison Results:")
        print("=" * 40)
        print(f"Custom model avg detections: {custom_avg:.2f}")
        print(f"Baseline model avg detections: {baseline_avg:.2f}")
        print(f"Improvement: {((custom_avg - baseline_avg) / baseline_avg * 100):.1f}%")
        print("=" * 40)

        return {
            'custom_avg': custom_avg,
            'baseline_avg': baseline_avg,
            'improvement_percent': (custom_avg - baseline_avg) / baseline_avg * 100
        }

    def full_evaluation(self):
        """Run complete evaluation suite"""
        print("🚀 Running Full Model Evaluation")
        print("=" * 50)

        results = {}

        print("\n1️⃣ Dataset Evaluation")
        dataset_metrics = self.evaluate_on_test_set()
        if dataset_metrics:
            results['dataset_metrics'] = dataset_metrics

        print("\n2️⃣ Confusion Matrix")
        self.create_confusion_matrix()

        print("\n3️⃣ Performance Testing")
        performance = self.test_inference_speed(50)
        if performance:
            results['performance'] = performance

        print("\n4️⃣ Baseline Comparison")
        comparison = self.compare_with_baseline()
        if comparison:
            results['comparison'] = comparison

        with open('outputs/complete_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n✅ Complete evaluation finished!")
        print("📁 Results saved in outputs/ folder")

        return results


def main():
    evaluator = ModelEvaluator()

    print("📊 Model Evaluation Tools")
    print("=" * 40)
    print("1. Evaluate on test dataset (mAP, precision, recall)")
    print("2. Create confusion matrix")
    print("3. Test inference speed")
    print("4. Compare with baseline YOLOv8")
    print("5. Run full evaluation suite")
    print("=" * 40)

    choice = input("Enter your choice (1-5): ").strip()

    if choice == "1":
        evaluator.evaluate_on_test_set()
    elif choice == "2":
        evaluator.create_confusion_matrix()
    elif choice == "3":
        evaluator.test_inference_speed()
    elif choice == "4":
        evaluator.compare_with_baseline()
    elif choice == "5":
        evaluator.full_evaluation()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Evaluation interrupted!")
    except Exception as e:
        print(f"❌ Error: {e}")
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import math
import os


class SimpleTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, class_id, confidence):
        """Register a new object"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'class_id': class_id,
            'confidence': confidence,
            'trail': [centroid]
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):

        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for detection in detections:
                centroid, class_id, confidence, _ = detection
                self.register(centroid, class_id, confidence)
        else:
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] -
                               np.array([det[0] for det in detections]), axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                centroid, class_id, confidence, _ = detections[col]

                self.objects[object_id]['centroid'] = centroid
                self.objects[object_id]['class_id'] = class_id
                self.objects[object_id]['confidence'] = confidence
                self.objects[object_id]['trail'].append(centroid)

                if len(self.objects[object_id]['trail']) > 10:
                    self.objects[object_id]['trail'].pop(0)

                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    centroid, class_id, confidence, _ = detections[col]
                    self.register(centroid, class_id, confidence)

        return self.objects


class ObjectTrackingSystem:
    def __init__(self, model_path="models/best.pt"):
        self.model_path = model_path
        self.model = None
        self.tracker = SimpleTracker(max_disappeared=20, max_distance=80)
        self.class_names = ['cat', 'dog', 'person']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                       (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Custom model loaded from {self.model_path}")
            else:
                self.model = YOLO('../animal-detector-backend/yolov8n.pt')
                print("⚠️ Using original YOLOv8 model (custom model not found)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = YOLO('../animal-detector-backend/yolov8n.pt')

    def get_detections(self, frame):
        if self.model is None:
            return []

        try:
            if "custom" in self.model_path or os.path.exists(self.model_path):
                results = self.model(frame, conf=0.4, verbose=False)
            else:
                results = self.model(frame, classes=[0, 15, 16], conf=0.4, verbose=False)

            detections = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        bbox = (x1, y1, x2, y2)

                        if not os.path.exists(self.model_path):
                            if class_id == 15:
                                class_id = 0
                            elif class_id == 16:
                                class_id = 1
                            elif class_id == 0:
                                class_id = 2
                            else:
                                continue

                        detections.append((centroid, class_id, confidence, bbox))

            return detections

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []

    def draw_tracking_info(self, frame, tracked_objects):
        for object_id, obj_info in tracked_objects.items():
            centroid = obj_info['centroid']
            class_id = obj_info['class_id']
            confidence = obj_info['confidence']
            trail = obj_info['trail']

            color = self.colors[object_id % len(self.colors)]

            if len(trail) > 1:
                for i in range(1, len(trail)):
                    thickness = max(1, int(3 * (i / len(trail))))  # Ensure thickness >= 1
                    cv2.line(frame, trail[i - 1], trail[i], color, thickness)

            cv2.circle(frame, centroid, 6, color, -1)
            cv2.circle(frame, centroid, 10, color, 2)

            class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
            label = f"ID:{object_id} {class_name} {confidence:.2f}"

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (centroid[0] - 5, centroid[1] - 25),
                          (centroid[0] + text_size[0] + 5, centroid[1] - 5), color, -1)
            cv2.putText(frame, label, (centroid[0], centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run_tracking(self, save_video=False):
        print("🎯 Starting Object Tracking")
        print("Controls:")
        print("- Q: Quit")
        print("- S: Save screenshot")
        print("- R: Reset tracker")
        print("=" * 40)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return

        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('outputs/tracking_demo.avi', fourcc, 20.0, (640, 480))

        frame_count = 0
        screenshot_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break

                frame_count += 1

                detections = self.get_detections(frame)

                tracked_objects = self.tracker.update(detections)

                self.draw_tracking_info(frame, tracked_objects)

                for detection in detections:
                    centroid, class_id, confidence, bbox = detection
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                info_text = f"Frame: {frame_count} | Objects: {len(tracked_objects)} | Detections: {len(detections)}"
                cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Object Tracking', frame)

                if video_writer:
                    video_writer.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'outputs/tracking_screenshot_{screenshot_count}.jpg', frame)
                    print(f"📸 Screenshot saved: tracking_screenshot_{screenshot_count}.jpg")
                    screenshot_count += 1
                elif key == ord('r'):
                    self.tracker = SimpleTracker(max_disappeared=20, max_distance=80)
                    print("🔄 Tracker reset")

        except KeyboardInterrupt:
            print("\n⏹️ Tracking stopped by user")

        finally:
            cap.release()
            if video_writer:
                video_writer.release()
                print("🎥 Video saved: outputs/tracking_demo.avi")
            cv2.destroyAllWindows()

            print(f"\n📊 Tracking Summary:")
            print(f"Total frames processed: {frame_count}")
            print(f"Final tracked objects: {len(self.tracker.objects)}")

    def analyze_tracking_performance(self, duration_seconds=30):
        print(f"📊 Analyzing tracking performance for {duration_seconds} seconds...")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return

        start_time = cv2.getTickCount()
        frame_count = 0
        tracking_stats = {
            'total_objects_tracked': 0,
            'max_simultaneous_objects': 0,
            'avg_objects_per_frame': 0,
            'tracking_accuracy': 0,
            'fps': 0
        }

        objects_per_frame = []
        max_objects = 0
        total_unique_objects = set()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = cv2.getTickCount()
                elapsed = (current_time - start_time) / cv2.getTickFrequency()

                if elapsed > duration_seconds:
                    break

                detections = self.get_detections(frame)
                tracked_objects = self.tracker.update(detections)

                num_objects = len(tracked_objects)
                objects_per_frame.append(num_objects)
                max_objects = max(max_objects, num_objects)
                total_unique_objects.update(tracked_objects.keys())

                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"Analyzed {frame_count} frames ({elapsed:.1f}s)")

        except KeyboardInterrupt:
            print("\n⏹️ Analysis stopped")

        finally:
            cap.release()

            total_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            tracking_stats['fps'] = frame_count / total_time if total_time > 0 else 0
            tracking_stats['total_objects_tracked'] = len(total_unique_objects)
            tracking_stats['max_simultaneous_objects'] = max_objects
            tracking_stats['avg_objects_per_frame'] = np.mean(objects_per_frame) if objects_per_frame else 0

            print("\n📈 Tracking Performance Results:")
            print("=" * 50)
            print(f"Total frames processed: {frame_count}")
            print(f"Total unique objects tracked: {tracking_stats['total_objects_tracked']}")
            print(f"Max simultaneous objects: {tracking_stats['max_simultaneous_objects']}")
            print(f"Average objects per frame: {tracking_stats['avg_objects_per_frame']:.2f}")
            print(f"Processing FPS: {tracking_stats['fps']:.2f}")
            print("=" * 50)

            return tracking_stats


def main():
    tracker_system = ObjectTrackingSystem()

    print("🎯 Object Tracking System")
    print("=" * 40)
    print("1. Run real-time tracking")
    print("2. Run tracking with video recording")
    print("3. Analyze tracking performance")
    print("=" * 40)

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        tracker_system.run_tracking(save_video=False)
    elif choice == "2":
        tracker_system.run_tracking(save_video=True)
    elif choice == "3":
        tracker_system.analyze_tracking_performance()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Tracking system stopped!")
    except Exception as e:
        print(f"❌ Error: {e}")
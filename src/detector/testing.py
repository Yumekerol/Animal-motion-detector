import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
class ModelTester:
    def __init__(self, dataset_handler):
        self.dataset_handler = dataset_handler

    def test_model_webcam(self, model_path="models/best11.pt"):
        if not os.path.exists(model_path):
            print("❌ Trained model not found!")
            print(f"Expected path: {model_path}")
            return

        print("🎥 Testing merged model with webcam...")
        print("📋 Classes that can be detected:")
        for i, name in enumerate(self.dataset_handler.class_names):
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

    def test_model_on_image(self, model_path="models/best11.pt"):
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
        for i, name in enumerate(self.dataset_handler.class_names):
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

    def test_model_on_video(self, model_path="models/best11.pt"):
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
        for i, name in enumerate(self.dataset_handler.class_names):
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
        class_counts = defaultdict(int)

        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if cls < len(self.dataset_handler.class_names):
                        class_name = self.dataset_handler.class_names[cls]
                    else:
                        class_name = f'Class_{cls}'

                    class_counts[class_name] += 1
                    detection_count += 1

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

        # Create pie chart visualization
        if class_counts:
            pie_chart_size = 150
            pie_chart = np.zeros((pie_chart_size, pie_chart_size, 3), dtype=np.uint8)

            total = sum(class_counts.values())
            start_angle = 0
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                      (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]

            for i, (class_name, count) in enumerate(class_counts.items()):
                angle = 360 * (count / total)
                color = colors[i % len(colors)]
                cv2.ellipse(pie_chart,
                            (pie_chart_size // 2, pie_chart_size // 2),
                            (pie_chart_size // 2 - 5, pie_chart_size // 2 - 5),
                            0, start_angle, start_angle + angle,
                            color, -1)
                start_angle += angle

            cv2.circle(pie_chart,
                       (pie_chart_size // 2, pie_chart_size // 2),
                       pie_chart_size // 4,
                       (255, 255, 255), -1)

            text = f"{total}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(pie_chart, text,
                        (pie_chart_size // 2 - text_size[0] // 2, pie_chart_size // 2 + text_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            annotated_frame[10:10 + pie_chart_size, -10 - pie_chart_size:-10] = pie_chart

        y_offset = 70
        cv2.putText(annotated_frame, f'Total Detections: {detection_count}',
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for i, (class_name, count) in enumerate(class_counts.items()):
            y_offset += 25
            cv2.putText(annotated_frame, f'{class_name}: {count}',
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated_frame
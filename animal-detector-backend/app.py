from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import time
import threading
from queue import Queue

from detect_animals2 import RoboflowAnimalDetector

app = Flask(__name__)
CORS(app)

training_progress = {
    'current_epoch': 0,
    'total_epochs': 0,
    'percentage': 0,
    'is_training': False,
    'message': ''
}

detector = RoboflowAnimalDetector()

model_cache = {}


def load_model_cached(model_path):
    if model_path not in model_cache:
        if os.path.exists(model_path):
            from ultralytics import YOLO
            model_cache[model_path] = YOLO(model_path)
        else:
            return None
    return model_cache[model_path]


@app.route('/api/test/webcam', methods=['POST'])
def test_webcam():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']

        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        height, width = image.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280 / width, 720 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        model_path = "models/best11.pt"
        model = load_model_cached(model_path)

        if model is None:
            return jsonify({'error': 'No trained model found'}), 400

        results = model(image, conf=0.25, verbose=False, imgsz=640)

        annotated_image = detector.draw_detections(image, results)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', annotated_image, encode_param)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        class_counts = {}
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls < len(detector.class_names):
                        class_name = detector.class_names[cls]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'classCounts': class_counts,
            'totalDetections': sum(class_counts.values())
        })

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/test/image', methods=['POST'])
def test_image():
    return test_webcam()


@app.route('/api/test/video', methods=['POST'])
def test_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400

        file = request.files['video']
        temp_video_path = f"temp_video_{int(time.time())}.mp4"
        file.save(temp_video_path)

        model_path = "models/best11.pt"
        model = load_model_cached(model_path)

        if model is None:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return jsonify({'error': 'No trained model found'}), 400

        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return jsonify({'error': 'Could not open video file'}), 400

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = min(total_frames, fps * 60)

        if width > 1280 or height > 720:
            scale = min(1280 / width, 720 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = width, height

        output_path = f"outputs/processed_video_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        class_counts = {}
        frame_count = 0
        skip_frames = max(1, fps // 5)

        print(f"Processando vídeo: {max_frames} frames, pulando {skip_frames} frames")

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            if new_width != width or new_height != height:
                frame = cv2.resize(frame, (new_width, new_height))

            results = model(frame, conf=0.3, verbose=False, imgsz=640)
            annotated_frame = detector.draw_detections(frame, results)

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls < len(detector.class_names):
                            class_name = detector.class_names[cls]
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for _ in range(skip_frames):
                out.write(annotated_frame)
                if frame_count >= max_frames:
                    break
                frame_count += 1

            if frame_count % (fps * 5) == 0:  # A cada 5 segundos
                progress = (frame_count / max_frames) * 100
                print(f"Progresso: {progress:.1f}%")

        cap.release()
        out.release()

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        cap = cv2.VideoCapture(output_path)
        ret, first_frame = cap.read()
        cap.release()

        img_base64 = None
        if ret:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', first_frame, encode_param)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}' if img_base64 else None,
            'classCounts': class_counts,
            'totalDetections': sum(class_counts.values()),
            'processedFrames': frame_count,
            'videoPath': output_path,
            'message': f'Processado {frame_count} frames de {total_frames} total'
        })

    except Exception as e:
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/test/video/progress', methods=['GET'])
def video_progress():
    return jsonify({'progress': 0})


@app.route('/api/evaluate', methods=['POST'])
def evaluate_model():
    try:
        model_path = "models/best11.pt"
        model = load_model_cached(model_path)

        if model is None:
            return jsonify({
                'success': False,
                'message': 'No trained model found'
            })

        if not detector.class_names:
            detector.load_dataset_config()

        detector.evaluate_model(model_path)

        return jsonify({
            'success': True,
            'message': 'Model evaluation completed! Check runs/detect/val/ for results.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Evaluation error: {str(e)}'
        })


@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        dataset_exists = os.path.exists(detector.merged_dataset_path)
        model_paths = ["models/best11.pt"]
        model_exists = any(os.path.exists(path) for path in model_paths)

        if dataset_exists and not detector.class_names:
            detector.load_dataset_config()

        return jsonify({
            'dataset_exists': dataset_exists,
            'model_exists': model_exists,
            'num_classes': detector.num_classes,
            'class_names': detector.class_names,
            'is_training': training_progress['is_training']
        })

    except Exception as e:
        return jsonify({
            'error': f'Status check error: {str(e)}'
        })


@app.route('/api/download/<path:filename>')
def download_file(filename):
    try:
        file_path = os.path.join('outputs', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("🚀 Animal Detector Backend starting...")
    print("📡 API endpoints available:")
    print("  - POST /api/test/webcam - Test with webcam")
    print("  - POST /api/test/image - Test with image")
    print("  - POST /api/test/video - Test with video")
    print("  - POST /api/evaluate - Evaluate model")
    print("  - GET  /api/status - Get system status")

    app.run(debug=True, host='0.0.0.0', port=5000)
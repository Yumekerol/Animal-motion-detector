from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import time

# Import your existing detector class
from detect_animals2 import RoboflowAnimalDetector

app = Flask(__name__)
CORS(app)

# Global variables for training progress
training_progress = {
    'current_epoch': 0,
    'total_epochs': 0,
    'percentage': 0,
    'is_training': False,
    'message': ''
}

detector = RoboflowAnimalDetector()


@app.route('/api/test/webcam', methods=['POST'])
def test_webcam():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']

        # Convert uploaded file to OpenCV image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Load model and make prediction
        model_path = "models/best11.pt"
        if not os.path.exists(model_path):
            # Try alternative paths
            alt_paths = ["models/best11.pt", "models/best11.pt"]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break

        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found'}), 400

        from ultralytics import YOLO
        model = YOLO(model_path)

        # Make prediction
        results = model(image, conf=0.3, verbose=False)

        # Draw detections
        annotated_image = detector.draw_detections(image, results)

        # Convert result to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Count detections by class
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
    return test_webcam()  # Same logic


@app.route('/api/test/video', methods=['POST'])
def test_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400

        file = request.files['video']

        # Save uploaded video temporarily
        temp_video_path = f"temp_video_{int(time.time())}.mp4"
        file.save(temp_video_path)

        # Load model
        model_path = "models/best11.pt"
        if not os.path.exists(model_path):
            alt_paths = ["models/best11.pt", "models/best11.pt"]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break

        if not os.path.exists(model_path):
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return jsonify({'error': 'No trained model found'}), 400

        from ultralytics import YOLO
        model = YOLO(model_path)

        # Process video
        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return jsonify({'error': 'Could not open video file'}), 400

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video
        output_path = f"outputs/processed_video_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        class_counts = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Make prediction
            results = model(frame, conf=0.3, verbose=False)

            # Draw detections
            annotated_frame = detector.draw_detections(frame, results)

            # Count detections
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls < len(detector.class_names):
                            class_name = detector.class_names[cls]
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            out.write(annotated_frame)
            frame_count += 1

            # Limit processing for demo (first 30 seconds)
            if frame_count > fps * 30:
                break

        cap.release()
        out.release()

        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        # For demo, return the first frame as image
        cap = cv2.VideoCapture(output_path)
        ret, first_frame = cap.read()
        cap.release()

        if ret:
            _, buffer = cv2.imencode('.jpg', first_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            img_base64 = None

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}' if img_base64 else None,
            'classCounts': class_counts,
            'totalDetections': sum(class_counts.values()),
            'processedFrames': frame_count,
            'videoPath': output_path
        })

    except Exception as e:
        # Clean up temp files
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate model performance"""
    try:
        model_path = "models/best11.pt"
        if not os.path.exists(model_path):
            alt_paths = ["models/best11.pt", "models/best11.pt"]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break

        if not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'message': 'No trained model found'
            })

        # Load dataset config if not already loaded
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
    """Get current system status"""
    try:
        # Check if dataset exists
        dataset_exists = os.path.exists(detector.merged_dataset_path)

        # Check if model exists
        model_paths = ["models/best11.pt", "models/best11.pt", "models/best11.pt"]
        model_exists = any(os.path.exists(path) for path in model_paths)

        # Load dataset info if available
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
    print("  - POST /api/datasets/download - Download and merge datasets")
    print("  - GET  /api/datasets/validate - Validate dataset")
    print("  - POST /api/train - Start training")
    print("  - GET  /api/training/progress - Get training progress")
    print("  - POST /api/test/webcam - Test with webcam")
    print("  - POST /api/test/image - Test with image")
    print("  - POST /api/test/video - Test with video")
    print("  - POST /api/evaluate - Evaluate model")
    print("  - GET  /api/status - Get system status")

    app.run(debug=True, host='0.0.0.0', port=5000)

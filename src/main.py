from ultralytics import YOLO
import cv2

print("🔄 Testing YOLO...")
model = YOLO('yolov8n.pt')
print("✅ YOLO installed and working!")

print("🔄 Testing webcam...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("✅ Webcam working!")
    cap.release()
else:
    print("❌ Problem with the webcam - using example video")

cv2.destroyAllWindows()
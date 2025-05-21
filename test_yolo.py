from ultralytics import YOLO
import cv2

# Testar instalação
print("🔄 A testar YOLO...")
model = YOLO('yolov8n.pt')  # Download automático na primeira vez
print("✅ YOLO instalado e a funcionar!")

# Testar webcam
print("🔄 A testar webcam...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("✅ Webcam a funcionar!")
    cap.release()
else:
    print("❌ Problema com webcam - vamos usar vídeo exemplo")

cv2.destroyAllWindows()
from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Use a fine-tuned model for car damage detection

# Initialize Webcam (Mobile Camera Stream)
def generate_frames():
    cap = cv2.VideoCapture(0)  # Use mobile camera if connected

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f"{box.cls[0]}: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

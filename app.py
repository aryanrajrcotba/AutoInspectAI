from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch
import pyttsx3
import threading
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Pre-trained YOLO model for person detection

# Initialize Webcam
cap = cv2.VideoCapture(0)  # Use webcam
# cap = cv2.VideoCapture("http://192.168.46.120:4747/video")  # Mobile Camera Stream

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

def play_alert():
    """Plays a voice alert when a person is too close."""
    engine.say("Warning! Person too close. Please stop! Stoooooppp")
    engine.runAndWait()

def detect_lanes(frame):
    """Detects lane lines using edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest = np.array([[
        (0, height), (width // 2, height // 2), (width, height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, region_of_interest, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow lane lines

    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect lanes
        frame = detect_lanes(frame)

        # Perform object detection (for detecting people)
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = float(box.conf[0])  # Confidence score
                class_id = int(box.cls[0])  # Class ID

                # Check if the detected object is a person (Class ID = 0 in YOLO)
                if class_id == 0:
                    width = x2 - x1
                    height = y2 - y1

                    # Define "too close" threshold based on bounding box size
                    if width * height > 50000:  # Adjust threshold based on testing
                        color = (0, 0, 255)  # Red box for "STOP"
                        label = "STOP! Person Too Close"

                        # Run voice alert in a separate thread (so it doesn't block video)
                        threading.Thread(target=play_alert).start()
                    else:
                        color = (0, 255, 0)  # Green box for normal detection
                        label = f"Person: {conf:.2f}"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
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

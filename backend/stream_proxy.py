from flask import Flask, Response, request, send_from_directory, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLOv8
import os
from twilio.rest import Client  # Import Twilio
import time

app = Flask(__name__)
CORS(app)  # Ensure this is applied to the Flask app
socketio = SocketIO(app)

# Update this to match your phone's IP camera URL
IP_CAMERA_URL = "http://192.168.2.42:8080/video"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a lightweight YOLOv8 model (e.g., yolov8n.pt)

# Twilio credentials
TWILIO_ACCOUNT_SID = "AC07b27799040c967676971358372abe34"  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = "8670ef1964fa4e0e51bc9673477e3e60"    # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "+14632836768"  # Replace with your Twilio phone number
TO_PHONE_NUMBER = "+919061722852"  # Replace with the recipient's phone number

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Alert cooldown to prevent spamming
last_alert_time = 0
ALERT_COOLDOWN = 60 # Minimum seconds between alerts

def send_sms_alert(num_persons):
    global last_alert_time
    current_time = time.time()

    # Check if enough time has passed since the last alert
    if current_time - last_alert_time >= ALERT_COOLDOWN:
        try:
            message = twilio_client.messages.create(
                body=f"⚠️ Alert: {num_persons} people detected in the monitoring area!",
                from_=TWILIO_PHONE_NUMBER,
                to=TO_PHONE_NUMBER
            )
            last_alert_time = current_time
            print(f"SMS alert sent successfully! SID: {message.sid}")
        except Exception as e:
            print(f"Error sending SMS alert: {e}")

def generate_frames(process_type="bounding"):
    print("generate_frames() function started.")
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap.isOpened():
        print("Error: Unable to connect to the IP camera.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Unable to read frame from the IP camera.")
            break

        # Resize the frame to reduce processing time
        frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 resolution

        # Run YOLOv8 model on the frame
        try:
            results = model(frame, conf=0.5, iou=0.4)  # Lower confidence and IoU thresholds for faster inference
            human_boxes = []
            for result in results[0].boxes:
                if int(result.cls) == 0:  # Class ID 0 corresponds to "person"
                    human_boxes.append(result.xyxy[0].cpu().numpy())

            # Trigger SMS alert if 3 or more people are detected
            if len(human_boxes) >= 3:
                send_sms_alert(len(human_boxes))

            if process_type == "bounding":
                for box in human_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                processed_frame = frame
            elif process_type == "heatmap":
                heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                for box in human_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    heatmap[y1:y2, x1:x2] += 1
                heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                processed_frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

    cap.release()

@app.route('/live-stream-bounding')
def live_stream_bounding():
    print("live_stream_bounding() endpoint called.")
    return Response(generate_frames(process_type="bounding"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live-stream-heatmap')
def live_stream_heatmap():
    print("live_stream_heatmap() endpoint called.")
    return Response(generate_frames(process_type="heatmap"), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Starting Flask server for live streaming...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
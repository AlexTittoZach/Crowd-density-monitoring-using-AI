from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__)

uploaded_video_path = None  # Global variable to store the path of the uploaded video

@app.route('/upload-video', methods=['POST'])
def upload_video():
    global uploaded_video_path
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files['video']
    upload_folder = "./uploaded_videos"
    os.makedirs(upload_folder, exist_ok=True)
    uploaded_video_path = os.path.join(upload_folder, video.filename)
    video.save(uploaded_video_path)
    print(f"Video saved: {uploaded_video_path}")
    return jsonify({"message": "Video uploaded successfully", "path": video.filename}), 200

@app.route('/process-video', methods=['POST'])
def process_video():
    global uploaded_video_path
    if not uploaded_video_path:
        return jsonify({"error": "No video uploaded"}), 400

    cap = cv2.VideoCapture(uploaded_video_path)
    motion_frames = []
    density_maps = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Example: Generate a dummy motion flow and density map
        motion_frame = cv2.Canny(frame, 100, 200)  # Edge detection as a placeholder for motion flow
        density_map = np.zeros_like(frame)
        density_map[:, :, 1] = 255  # Green channel as a placeholder for density map

        motion_frames.append(motion_frame)
        density_maps.append(density_map)

    cap.release()
    return jsonify({"message": "Video processed successfully", "frames": len(motion_frames)}), 200

@app.route('/static/<path:filename>')
def serve_uploaded_video(filename):
    upload_folder = "./uploaded_videos"
    return send_from_directory(upload_folder, filename)

if __name__ == "__main__":
    print("Starting Flask server for video uploading...")
    app.run(host="0.0.0.0", port=5001, debug=True)

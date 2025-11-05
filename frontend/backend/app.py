from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import List
import asyncio
from datetime import datetime
import json
from crowd_processor import CrowdDensityDetector
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
active_connections: List[WebSocket] = []

class FallDetector:
    def __init__(self):
        # Initialize your AI model here
        # self.model = load_model()
        pass

    async def process_frame(self, frame):
        # Implement fall detection logic
        # 1. Preprocess frame
        # 2. Run through model
        # 3. Post-process results
        pass

fall_detector = FallDetector()
crowd_detector = CrowdDensityDetector()

@app.websocket("/ws/fall-detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame
            results = await fall_detector.process_frame(frame)
            
            # Send results back to client
            await websocket.send_json({
                "timestamp": datetime.now().isoformat(),
                "fall_detected": results.get("fall_detected", False),
                "confidence": results.get("confidence", 0),
                "pose_keypoints": results.get("pose_keypoints", [])
            })
    except Exception as e:
        print(f"Error: {e}")
    finally:
        active_connections.remove(websocket)

@app.websocket("/ws/crowd-detection")
async def crowd_detection_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                # Receive frame from client
                data = await websocket.receive_bytes()
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process frame
                motion_frame, density_map = await crowd_detector.process_frame(frame)
                
                # Encode frames to JPEG
                _, motion_encoded = cv2.imencode('.jpg', motion_frame)
                _, density_encoded = cv2.imencode('.jpg', density_map)
                
                # Convert to base64 for JSON transfer
                motion_b64 = base64.b64encode(motion_encoded.tobytes()).decode('utf-8')
                density_b64 = base64.b64encode(density_encoded.tobytes()).decode('utf-8')
                
                # Send results back to client
                await websocket.send_json({
                    "timestamp": datetime.now().isoformat(),
                    "motion_frame": motion_b64,
                    "density_map": density_b64
                })
            except Exception as frame_error:
                print(f"Error processing frame: {frame_error}")
                # Continue to next frame instead of breaking connection
                await asyncio.sleep(0.1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Ensure connection is closed properly
        try:
            await websocket.close()
        except:
            pass

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())
    
    # Process the video
    cap = cv2.VideoCapture(file_location)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        motion_frame, density_map = await crowd_detector.process_frame(frame)
        frames.append({
            "motion": motion_frame,
            "density": density_map
        })
    
    cap.release()
    
    # Return processed frames
    return {"message": "Video processed successfully", "frame_count": len(frames)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
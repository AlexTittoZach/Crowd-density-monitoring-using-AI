import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Optional
import time

class MotionAnalyzer:
    def __init__(self, frame_shape: Tuple[int, int]):
        self.height, self.width = frame_shape
        self.prev_gray = None
        self.prev_points = None
        self.tracks = []
        self.max_tracks = 50
        self.track_len = 30
        self.detect_interval = 5
        self.frame_count = 0
        self.flow_visualization = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stagnant_groups = []
        self.stagnant_threshold = 8
        self.gathering_radius = 40
        self.min_group_size = 2
        
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.color = np.random.randint(0, 255, (100, 3))
        self.stagnant_points = []
        self.stagnant_counts = []

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.tracks = []
        self.stagnant_points = []
        self.stagnant_counts = []

class CrowdDensityDetector:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.motion_analyzer = None
        
    def initialize_motion_analyzer(self, frame_shape):
        if self.motion_analyzer is None:
            self.motion_analyzer = MotionAnalyzer(frame_shape)
    
    async def process_frame(self, frame):
        if frame is None:
            return None, None
            
        # Initialize motion analyzer if not already done
        if self.motion_analyzer is None:
            self.initialize_motion_analyzer((frame.shape[0], frame.shape[1]))
            
        # Run YOLO detection
        results = self.model(frame, classes=[0])  # class 0 is person
        
        # Create motion flow visualization
        motion_frame = frame.copy()
        
        # Create density map visualization
        density_map = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        density_map.fill(255)  # White background
        
        # Process detections
        if len(results) > 0:
            boxes = results[0].boxes
            if len(boxes) > 0:
                # Get all person detections
                person_boxes = boxes.xyxy.cpu().numpy()
                
                # Draw density circles
                for i in range(len(person_boxes)):
                    x1, y1, x2, y2 = person_boxes[i]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Draw circle on density map
                    cv2.circle(density_map, (center_x, center_y), 30, (255, 200, 100), -1)
                    
                # Find dense areas (areas with overlapping circles)
                for i in range(len(person_boxes)):
                    x1, y1, x2, y2 = person_boxes[i]
                    center1 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    # Count nearby people
                    nearby_count = 0
                    for j in range(len(person_boxes)):
                        if i != j:
                            x1_2, y1_2, x2_2, y2_2 = person_boxes[j]
                            center2 = (int((x1_2 + x2_2) / 2), int((y1_2 + y2_2) / 2))
                            
                            # Calculate distance between centers
                            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                            if distance < 100:  # Distance threshold for density
                                nearby_count += 1
                    
                    # If dense area found, draw annotation
                    if nearby_count >= 1:
                        cv2.circle(density_map, center1, 50, (0, 0, 255), 2)
                        cv2.putText(density_map, f"DENSE AREA: {nearby_count + 1} people",
                                  (center1[0] - 70, center1[1] - 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return motion_frame, density_map 
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Optional
import time

class MotionAnalyzer:
    """Analyzes motion patterns using Lucas-Kanade optical flow algorithm."""
    
    def __init__(self, frame_shape: Tuple[int, int]):
        self.height, self.width = frame_shape
        self.prev_gray = None
        self.prev_points = None
        self.tracks = []  # Store motion tracks
        self.max_tracks = 50  # Maximum number of tracks to display
        self.track_len = 30  # Increased from 15 to 30 for much longer tracks
        self.detect_interval = 5  # Detect new features every 5 frames
        self.frame_count = 0
        self.flow_visualization = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stagnant_groups = []  # Track potential gatherings
        self.stagnant_threshold = 8  # Increased: Frames to consider a point stagnant
        self.gathering_radius = 40  # Reduced: Radius to consider points as a group
        self.min_group_size = 2  # Reduced: Minimum number of people to consider a gathering
        
        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Colors for visualization
        self.color = np.random.randint(0, 255, (100, 3))
        
        # Track stagnant points
        self.stagnant_points = []  # List of points that haven't moved significantly
        self.stagnant_counts = []  # How long each point has been stagnant
    
    def reset(self):
        """Reset the analyzer state."""
        self.prev_gray = None
        self.prev_points = None
        self.tracks = []
        self.frame_count = 0
        self.flow_visualization = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.stagnant_points = []
        self.stagnant_counts = []
        self.stagnant_groups = []
    
    def update(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Update motion analysis with the new frame.
        
        Args:
            frame: Current video frame
            boxes: List of bounding boxes from person detection
            
        Returns:
            Flow visualization frame
        """
        self.frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        
        # Create a mask based on person detections to focus motion analysis on people
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            # Expand the box slightly to catch motion at the edges
            x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
            x2, y2 = min(self.width, x2 + 5), min(self.height, y2 + 5)
            mask[y1:y2, x1:x2] = 255
        
        # Initialize tracking on first frame
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return vis
            
        # Process optical flow
        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            
            # Update stagnant points tracking
            new_stagnant_points = []
            new_stagnant_counts = []
            
            for i, (tr, (x, y), good_flag) in enumerate(zip(self.tracks, p1.reshape(-1, 2), good)):
                if not good_flag:
                    continue
                
                # Ensure coordinates are within image boundaries
                x = max(0, min(int(x), self.width - 1))
                y = max(0, min(int(y), self.height - 1))
                
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                
                # Check if this point is stagnant (not moving much)
                if len(tr) > 1:
                    # Calculate movement distance
                    movement = np.sqrt((tr[-1][0] - tr[-2][0])**2 + (tr[-1][1] - tr[-2][1])**2)
                    
                    # If movement is below threshold, consider it stagnant
                    if movement < 2.0:  # Threshold for stagnant detection
                        if i < len(self.stagnant_counts):
                            new_stagnant_points.append((x, y))
                            new_stagnant_counts.append(self.stagnant_counts[i] + 1)
                        else:
                            new_stagnant_points.append((x, y))
                            new_stagnant_counts.append(1)
                    else:
                        new_stagnant_points.append((x, y))
                        new_stagnant_counts.append(0)
                else:
                    new_stagnant_points.append((x, y))
                    new_stagnant_counts.append(0)
                
                # Draw points with different colors based on movement
                if i < len(self.stagnant_counts) and self.stagnant_counts[i] > self.stagnant_threshold:
                    # Stagnant point - draw in red
                    cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
                else:
                    # Moving point - draw in green
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            
            self.tracks = new_tracks
            self.stagnant_points = new_stagnant_points
            self.stagnant_counts = new_stagnant_counts
            
            # Draw motion tracks with increased thickness for better visibility
            # Use a gradient color effect for tracks to show history
            for tr in self.tracks:
                if len(tr) > 1:
                    # Draw track history with a thickness gradient (thicker to thinner)
                    for i in range(len(tr)-1):
                        # Calculate alpha based on position in track (newer points = more opaque)
                        alpha = 0.5 + 0.5 * (i / (len(tr) - 1))
                        
                        # Thickness decreases for older points
                        thickness = max(1, int(3 * (i / (len(tr) - 1))))
                        
                        # Draw line segment with gradient thickness
                        cv2.line(
                            vis,
                            tr[i],
                            tr[i+1],
                            (0, int(255 * alpha), 0),
                            thickness
                        )
            
            # Draw motion vectors with increased length and thickness
            for tr in self.tracks:
                if len(tr) > 1:
                    # Calculate vector direction
                    dx = tr[-1][0] - tr[-2][0]
                    dy = tr[-1][1] - tr[-2][1]
                    
                    # Extend the vector for better visibility (3x longer)
                    end_x = int(tr[-1][0] + dx * 2.0)
                    end_y = int(tr[-1][1] + dy * 2.0)
                    
                    # Ensure end point is within image boundaries
                    end_x = max(0, min(end_x, self.width - 1))
                    end_y = max(0, min(end_y, self.height - 1))
                    
                    # Draw the arrowed line with increased thickness
                    cv2.arrowedLine(vis, tr[-1], (end_x, end_y), (0, 0, 255), 3, tipLength=0.5)
            
            # Detect gatherings of stagnant points
            self.stagnant_groups = []
            processed = set()
            
            # Create a mapping from stagnant points to actual person boxes
            point_to_box = {}
            for i, (pt, count) in enumerate(zip(self.stagnant_points, self.stagnant_counts)):
                if count < self.stagnant_threshold:
                    continue
                    
                # Find the closest person box to this point
                min_dist = float('inf')
                closest_box = None
                for box in boxes:
                    box_center_x = (box[0] + box[2]) // 2
                    box_center_y = (box[1] + box[3]) // 2
                    dist = np.sqrt((pt[0] - box_center_x)**2 + (pt[1] - box_center_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_box = box
                
                if min_dist < 50:  # Only associate if point is close to a box
                    point_to_box[i] = closest_box
            
            for i, (pt, count) in enumerate(zip(self.stagnant_points, self.stagnant_counts)):
                if i in processed or count < self.stagnant_threshold:
                    continue
                
                # Start a new group with this point
                group = [pt]
                group_boxes = set()
                if i in point_to_box:
                    group_boxes.add(tuple(point_to_box[i]))
                processed.add(i)
                
                # Find nearby stagnant points
                for j, (pt2, count2) in enumerate(zip(self.stagnant_points, self.stagnant_counts)):
                    if j in processed or count2 < self.stagnant_threshold:
                        continue
                    
                    # Calculate distance
                    dist = np.sqrt((pt[0] - pt2[0])**2 + (pt[1] - pt2[1])**2)
                    if dist < self.gathering_radius:
                        group.append(pt2)
                        processed.add(j)
                        if j in point_to_box:
                            group_boxes.add(tuple(point_to_box[j]))
                
                # Count unique boxes in the group
                person_count = len(group_boxes)
                
                # If group is large enough, add to stagnant groups
                if person_count >= self.min_group_size:
                    self.stagnant_groups.append({
                        'points': group,
                        'person_count': person_count
                    })
            
            # Visualize gatherings
            for group in self.stagnant_groups:
                points = group['points']
                person_count = group['person_count']
                
                # Calculate center of the group
                center_x = int(sum(p[0] for p in points) / len(points))
                center_y = int(sum(p[1] for p in points) / len(points))
                
                # Calculate radius based on group spread
                max_dist = max(np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in points)
                radius = max(int(max_dist), self.gathering_radius)
                
                # Draw circle around the gathering
                cv2.circle(vis, (center_x, center_y), radius, (0, 0, 255), 2)
                
                # Add alert text
                cv2.putText(
                    vis,
                    f'GATHERING: {person_count} people',
                    (center_x - radius, center_y - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
        
        # Detect new features periodically
        if self.frame_count % self.detect_interval == 0:
            # Draw ROI based on detected boxes
            for x1, y1, x2, y2 in boxes:
                # Ensure coordinates are within image boundaries
                x1 = max(0, min(x1, self.width - 1))
                y1 = max(0, min(y1, self.height - 1))
                x2 = max(0, min(x2, self.width - 1))
                y2 = max(0, min(y2, self.height - 1))
                
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
            # Detect good features to track within the ROI
            for x1, y1, x2, y2 in boxes:
                # Ensure coordinates are within image boundaries
                x1 = max(0, min(x1, self.width - 1))
                y1 = max(0, min(y1, self.height - 1))
                x2 = max(0, min(x2, self.width - 1))
                y2 = max(0, min(y2, self.height - 1))
                
                roi = frame_gray[y1:y2, x1:x2]
                if roi.size == 0:  # Skip empty ROIs
                    continue
                    
                roi_corners = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
                if roi_corners is not None:
                    roi_corners[:, 0, 0] += x1
                    roi_corners[:, 0, 1] += y1
                    for x, y in roi_corners.reshape(-1, 2):
                        # Ensure coordinates are within image boundaries
                        x = max(0, min(int(x), self.width - 1))
                        y = max(0, min(int(y), self.height - 1))
                        
                        # Check if the point is within the mask
                        if mask[y, x] > 0:
                            self.tracks.append([(x, y)])
                            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        
        # Limit the number of tracks
        if len(self.tracks) > self.max_tracks:
            self.tracks = self.tracks[-self.max_tracks:]
            
        # Add visualization text
        cv2.putText(
            vis,
            'Motion Flow',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Update for next frame
        self.prev_gray = frame_gray.copy()
        return vis

class DensityVisualizer:
    def __init__(self, frame_shape: Tuple[int, int], kernel_size: int = 35):
        self.height, self.width = frame_shape
        self.kernel_size = kernel_size
        self.density_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.gathering_threshold = 0.75  # Increased threshold for gathering detection
        self.min_gathering_area = 600  # Increased minimum area to consider as a gathering
        self.gathering_regions = []  # Store detected gathering regions
        self.person_boxes = []  # Store current person boxes for accurate counting
    
    def update_density(self, boxes: List[Tuple[int, int, int, int]], decay: float = 0.80):
        # Store current boxes for counting
        self.person_boxes = boxes
        
        # Apply decay to previous values
        self.density_map *= decay
        
        # Add new detections
        for x1, y1, x2, y2 in boxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            y, x = np.ogrid[-self.kernel_size:self.kernel_size + 1, 
                           -self.kernel_size:self.kernel_size + 1]
            mask = x*x + y*y <= self.kernel_size*self.kernel_size
            kernel = np.zeros((2*self.kernel_size + 1, 2*self.kernel_size + 1))
            kernel[mask] = 1
            kernel = cv2.GaussianBlur(kernel, (11, 11), 0)
            kernel = kernel / kernel.max()
            kernel *= 1.5
            
            y_min = max(0, center_y - self.kernel_size)
            y_max = min(self.height, center_y + self.kernel_size + 1)
            x_min = max(0, center_x - self.kernel_size)
            x_max = min(self.width, center_x + self.kernel_size + 1)
            
            kernel_y_min = max(0, self.kernel_size - center_y)
            kernel_y_max = kernel_y_min + (y_max - y_min)
            kernel_x_min = max(0, self.kernel_size - center_x)
            kernel_x_max = kernel_x_min + (x_max - x_min)
            
            self.density_map[y_min:y_max, x_min:x_max] += \
                kernel[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]
        
        # Detect gathering regions
        self.detect_gatherings()
    
    def detect_gatherings(self):
        # Normalize density map for threshold comparison
        normalized = self.density_map / (self.density_map.max() + 1e-10)
        
        # Create binary mask of high-density regions
        high_density = (normalized > self.gathering_threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        high_density = cv2.morphologyEx(high_density, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of high-density regions
        contours, _ = cv2.findContours(high_density, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Instead of using contours, directly check proximity between people
        self.gathering_regions = []
        
        # Group people by proximity
        processed = set()
        for i, box1 in enumerate(self.person_boxes):
            if i in processed:
                continue
            
            box1_center = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
            group = [box1]
            processed.add(i)
            
            for j, box2 in enumerate(self.person_boxes):
                if j in processed or i == j:
                    continue
                
                box2_center = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
                distance = np.sqrt((box1_center[0] - box2_center[0])**2 + (box1_center[1] - box2_center[1])**2)
                
                # If people are close enough, add to group
                if distance < 100:  # Adjust this threshold as needed
                    group.append(box2)
                    processed.add(j)
            
            # If group is large enough, consider it a gathering
            if len(group) >= 2:
                # Calculate center and radius
                centers_x = [(box[0] + box[2]) // 2 for box in group]
                centers_y = [(box[1] + box[3]) // 2 for box in group]
                center_x = sum(centers_x) // len(centers_x)
                center_y = sum(centers_y) // len(centers_y)
                
                # Calculate radius based on spread
                max_dist = 0
                for box in group:
                    box_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                    dist = np.sqrt((center_x - box_center[0])**2 + (center_y - box_center[1])**2)
                    max_dist = max(max_dist, dist)
                
                radius = int(max_dist + 50)  # Add padding
                
                self.gathering_regions.append({
                    'center': (center_x, center_y),
                    'radius': radius,
                    'area': np.pi * radius * radius,
                    'person_count': len(group)
                })
    
    def get_density_visualization(self, frame: np.ndarray) -> np.ndarray:
        # Normalize density map
        normalized = self.density_map / (self.density_map.max() + 1e-10)
        normalized = np.power(normalized, 0.5)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Create white background
        background = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        
        # Blend heatmap with background
        alpha = (normalized * 0.6)
        blended = np.zeros_like(heatmap)
        for i in range(3):
            blended[:, :, i] = heatmap[:, :, i] * alpha + background[:, :, i] * (1 - alpha)
        
        # Draw gathering regions
        for region in self.gathering_regions:
            center = region['center']
            radius = region['radius']
            person_count = region['person_count']
            
            # Draw circle around gathering
            cv2.circle(blended, center, radius, (0, 0, 255), 2)
            
            # Add alert text with accurate person count
            cv2.putText(
                blended,
                f'DENSE AREA: {person_count} people',
                (center[0] - radius, center[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Add title
        cv2.putText(
            blended,
            'Density Map',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        
        return blended.astype(np.uint8)

class FrameProcessor:
    """Processes video frames for motion analysis and density visualization."""
    
    def __init__(self, input_path: str):
        """
        Initialize the frame processor.
        
        Args:
            input_path: Path to the input video file
        """
        self.input_path = input_path
        self.cap = cv2.VideoCapture(input_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize YOLO model - upgrade from YOLOv8n to YOLOv8s
        self.model = YOLO('yolov8s.pt')
        
        # Initialize motion analyzer
        self.motion_analyzer = MotionAnalyzer((self.height, self.width))
        
        # Initialize density visualizer
        self.density_visualizer = DensityVisualizer((self.height, self.width))
        
        # Frame counter
        self.frame_count = 0
        
        self.view_mode = 0  # 0: Motion+Density, 1: Detection only, 2: Motion only, 3: Density only
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
        if self.density_visualizer is None:
            self.density_visualizer = DensityVisualizer(frame.shape[:2])
            
        if self.motion_analyzer is None:
            self.motion_analyzer = MotionAnalyzer(frame.shape[:2])
        
        results = self.model(frame, classes=[0])
        boxes = []
        detection_frame = frame.copy()
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                if confidence > 0.5:
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    cv2.rectangle(
                        detection_frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 
                        2
                    )
        
        count = len(boxes)
        
        # Update density visualization
        self.density_visualizer.update_density(boxes)
        density_frame = self.density_visualizer.get_density_visualization(frame)
        
        # Update motion analysis
        motion_frame = self.motion_analyzer.update(frame, boxes)
        
        # Fix title positions to avoid overlap
        # Add count to detection frame
        cv2.putText(
            detection_frame, 
            'Person Detection', 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        cv2.putText(
            detection_frame, 
            f'Count: {count}', 
            (20, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return detection_frame, density_frame, motion_frame, boxes

    def process_video(self, output_path: str):
        """
        Process the entire video and save the result.
        
        Args:
            output_path: Path to save the processed video
        """
        from pathlib import Path
        
        # Convert string path to Path object
        input_path = Path(self.input_path)
        
        # Check if input is a video file
        is_video_file = input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
        
        if is_video_file:
            # Process video file
            cap = cv2.VideoCapture(str(self.input_path))
            if not cap.isOpened():
                raise ValueError(f"Unable to open video file: {self.input_path}")
                
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default to 30 fps if unable to determine
            
            # Create a frame generator from video
            def frame_generator():
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame
                cap.release()
                
            frames = frame_generator()
        else:
            # Process directory of frames
            frame_paths = sorted(self.input_path.glob('*.jpg'))
            if not frame_paths:
                raise ValueError("No frames found in specified directory")
                
            first_frame = cv2.imread(str(frame_paths[0]))
            if first_frame is None:
                raise ValueError("Unable to read the first frame")
                
            frame_height, frame_width = first_frame.shape[:2]
            fps = 30  # Default fps for frame sequences
            
            # Create a frame generator from image files
            def frame_generator():
                for frame_path in frame_paths:
                    frame = cv2.imread(str(frame_path))
                    if frame is not None:
                        yield frame
                        
            frames = frame_generator()
        
        # Initialize visualizers early
        if self.density_visualizer is None:
            self.density_visualizer = DensityVisualizer((frame_height, frame_width))
            
        if self.motion_analyzer is None:
            self.motion_analyzer = MotionAnalyzer((frame_height, frame_width))
        
        # Create a consistent window with the specified size
        window_title = 'Crowd Analysis'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        
        # Set fixed window dimensions as requested
        display_width = 1250  # Fixed width as requested
        display_height = 550  # Fixed height as requested
        
        # Set initial window size
        cv2.resizeWindow(window_title, display_width, display_height)
        
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,  # Use the detected fps
                (frame_width * 2, frame_height)  # Double width for side-by-side display
            )
        
        try:
            decay = 0.80
            intensity_factor = 1.5
            frame_time = 1.0 / fps  # Time per frame in seconds
            
            for frame in frames:
                start_time = time.time()  # Start timing for this frame
                
                if frame is None:
                    continue
                    
                # Update kernel size dynamically for DensityVisualizer
                self.density_visualizer.kernel_size = int(intensity_factor * 35)
                
                # Process frame
                detection_frame, density_frame, motion_frame, boxes = self.process_frame(frame)
                
                # Updated view modes:
                # 0: Motion + Density (side by side)
                # 1: Detection only (full screen)
                # 2: Motion only (full screen)
                # 3: Density only (full screen)
                if self.view_mode == 0:
                    # Motion + Density (side by side)
                    display_frame = np.hstack((motion_frame, density_frame))
                    mode_name = "Motion and Density"
                elif self.view_mode == 1:
                    # Detection only (full screen)
                    display_frame = detection_frame
                    mode_name = "Person Detection"
                elif self.view_mode == 2:
                    # Motion only (full screen)
                    display_frame = motion_frame
                    mode_name = "Motion Flow"
                else:  # view_mode == 3
                    # Density only (full screen)
                    display_frame = density_frame
                    mode_name = "Density Map"
                
                # Update window title with current mode
                cv2.setWindowTitle(window_title, f'Crowd Analysis - {mode_name}')
                
                # Add a legend to the frame
                legend = [
                    "+: Increase decay (slower fade)",
                    "-: Decrease decay (faster fade)",
                    "i: Increase intensity",
                    "d: Decrease intensity",
                    "v: Change view mode",
                    "r: Reset motion tracking",
                    "q: Quit process",
                ]
                y_offset = 20
                for i, text in enumerate(legend):
                    y_pos = frame_height - (y_offset * (len(legend) - i))
                    
                    # Adjust position for different view modes
                    if self.view_mode == 0:
                        x_pos = 20  # For side-by-side view
                    else:
                        x_pos = 20  # For full screen views
                    
                    cv2.putText(
                        display_frame,
                        text,
                        (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                
                # Resize window if needed for different view modes
                if self.view_mode == 0:
                    # Side-by-side view (fixed width)
                    cv2.resizeWindow(window_title, display_width, display_height)
                else:
                    # Single view (same fixed width)
                    cv2.resizeWindow(window_title, display_width, display_height)
                
                cv2.imshow(window_title, display_frame)
                
                # Calculate how long to wait to maintain correct playback speed
                processing_time = time.time() - start_time
                wait_time = max(1, int((frame_time - processing_time) * 1000))
                
                # Adjust parameters dynamically
                key = cv2.waitKey(wait_time)
                if key == ord('+'):
                    decay = min(decay + 0.01, 1.0)  # Increase decay (slower fade)
                elif key == ord('-'):
                    decay = max(decay - 0.01, 0.5)  # Decrease decay (faster fade)
                elif key == ord('i'):  # Increase intensity
                    intensity_factor = min(intensity_factor + 0.1, 3.0)
                    print(f"Intensity factor increased to {intensity_factor}")
                elif key == ord('d'):  # Decrease intensity
                    intensity_factor = max(intensity_factor - 0.1, 0.5)
                    print(f"Intensity factor decreased to {intensity_factor}")
                elif key == ord('v'):  # Change view mode
                    self.view_mode = (self.view_mode + 1) % 4
                    mode_names = ["Motion + Density", "Detection only", "Motion only", "Density only"]
                    print(f"View mode changed to: {mode_names[self.view_mode]}")
                elif key == ord('r'):  # Reset motion tracking
                    self.motion_analyzer.reset()
                    print("Motion tracking reset")
                
                self.density_visualizer.update_density(boxes, decay=decay)
                
                if writer and self.view_mode == 0:  # Only save side-by-side view
                    writer.write(display_frame)
                elif writer:  # For single views, create a side-by-side duplicate for consistent video output
                    video_frame = np.hstack((display_frame, display_frame))
                    writer.write(video_frame)
                
                if key == ord('q'):
                    break
                
        finally:
            cv2.destroyAllWindows()
            if writer:
                writer.release()
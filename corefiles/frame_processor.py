import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple

class DensityVisualizer:
    def __init__(self, frame_shape: Tuple[int, int], kernel_size: int = 35):  # Reduced from 50 to 35
        self.height, self.width = frame_shape
        self.kernel_size = kernel_size
        self.density_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.scene_outline = None
        
    
    def update_density(self, boxes: List[Tuple[int, int, int, int]], decay: float = 0.80):
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
            kernel = cv2.GaussianBlur(kernel, (11, 11), 0)  # Reduced from (15, 15) to (11, 11)
            kernel = kernel / kernel.max()
            kernel *= 1.5  # Reduced from 2.0 to 1.5 for more concentrated density
            
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
    
    def get_density_visualization(self, frame: np.ndarray) -> np.ndarray:
        
        
        normalized = self.density_map / (self.density_map.max() + 1e-10)
        normalized = np.power(normalized, 0.5)
        
        heatmap = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        background = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        
        
        alpha = (normalized * 0.6)
        blended = np.zeros_like(heatmap)
        for i in range(3):
            blended[:, :, i] = heatmap[:, :, i] * alpha + background[:, :, i] * (1 - alpha)
        
        cv2.putText(
            blended,
            'Density Map',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        
        return blended
class FrameProcessor:
    def __init__(self, frames_path: str):
        self.frames_path = Path(frames_path)
        self.model = YOLO('yolov8n.pt')
        self.density_visualizer = None
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
        if self.density_visualizer is None:
            self.density_visualizer = DensityVisualizer(frame.shape[:2])
        
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
        cv2.putText(
            detection_frame, 
            f'Count: {count}', 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        self.density_visualizer.update_density(boxes)
        density_frame = self.density_visualizer.get_density_visualization(frame)
        
        return detection_frame, density_frame, boxes
    
    def process_video(self, output_path: str = None):
        frame_paths = sorted(self.frames_path.glob('*.jpg'))
        if not frame_paths:
            raise ValueError("No frames found in specified directory")
            
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise ValueError("Unable to read the first frame")
            
        frame_height, frame_width = first_frame.shape[:2]
        
        # Initialize DensityVisualizer early
        if self.density_visualizer is None:
            self.density_visualizer = DensityVisualizer((frame_height, frame_width))
        
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                10,  # fps
                (frame_width * 2, frame_height)  # Double width for side-by-side display
            )
        
        try:
            decay = 0.80
            intensity_factor = 1.5  # Start with default intensity factor
            
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                    
                # Update kernel size dynamically for DensityVisualizer
                self.density_visualizer.kernel_size = int(intensity_factor * 35)
                
                detection_frame, density_frame, boxes = self.process_frame(frame)
                combined_frame = np.hstack((detection_frame, density_frame))
                # Add a legend to the frame
                legend = [
                    "+: Increase decay (slower fade)",
                    "-: Decrease decay (faster fade)",
                    "i: Increase intensity",
                    "d: Decrease intensity",
                    "q: Quit process",
                ]
                y_offset = 20
                for i, text in enumerate(legend):
                    cv2.putText(
                        combined_frame,
                        text,
                        (20, frame_height - (y_offset * (len(legend) - i))),  # Position legend at bottom-left
                        cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                
                cv2.imshow('Crowd Flow Analysis: A mini-project', combined_frame)
                
                # Adjust intensity and decay dynamically
                key = cv2.waitKey(1)
                if key == ord('+'):
                    decay = min(decay + 0.01, 1.0)  # Increase decay (slower fade)
                elif key == ord('-'):
                    decay = max(decay - 0.01, 0.5)  # Decrease decay (faster fade)
                elif key == ord('i'):  # Increase intensity
                    intensity_factor = min(intensity_factor + 0.1, 3.0)  # Limit max intensity
                    print(f"Intensity factor increased to {intensity_factor}")
                elif key == ord('d'):  # Decrease intensity
                    intensity_factor = max(intensity_factor - 0.1, 0.5)  # Limit min intensity
                    print(f"Intensity factor decreased to {intensity_factor}")
                
                self.density_visualizer.update_density(boxes, decay=decay)
                
                if writer:
                    writer.write(combined_frame)
                
                if key == ord('q'):
                    break
                    
        finally:
            cv2.destroyAllWindows()
            if writer:
                writer.release()


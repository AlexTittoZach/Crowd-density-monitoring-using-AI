import cv2
import numpy as np
import matplotlib as mpl  # Import matplotlib base package
import matplotlib.pyplot as plt
from ultralytics import YOLO

def test_setup():
    print("Testing library imports:")
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    print("Matplotlib version:", mpl.__version__)  # Use mpl instead of plt
    
    # Test YOLO loading
    try:
        model = YOLO('yolov8s.pt')
        print("YOLO model loaded successfully")
    except Exception as e:
        print("Error loading YOLO:", e)

if __name__ == "__main__":
    test_setup()
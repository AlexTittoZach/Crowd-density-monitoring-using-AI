import argparse
from frame_processor3 import FrameProcessor

def main():
    parser = argparse.ArgumentParser(description='Process video or frames for crowd analysis')
    parser.add_argument('input_path', type=str, help='C:/Users/ronro/OneDrive/Desktop/miniproject1/input_video2.mp4')
    parser.add_argument('--output', type=str, help='Path to output video file (optional)')
    
    args = parser.parse_args()
    
    processor = FrameProcessor(args.input_path)
    processor.process_video(args.output)

if __name__ == '__main__':
    main()
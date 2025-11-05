import cv2

# Replace with the IP address provided by your phone's camera app
IP_CAMERA_URL = "http://192.168.2.42:8080/video"

# Load a pre-trained Haar cascade for people detection
PEOPLE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def detect_crowd(frame):
    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect people in the frame
    people = PEOPLE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    # Draw rectangles around detected people
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def main():
    print("Connecting to IP camera...")
    cap = cv2.VideoCapture(IP_CAMERA_URL)

    if not cap.isOpened():
        print("Error: Unable to connect to the IP camera. Check the URL or network connection.")
        return

    print("Connected to IP camera. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the IP camera.")
            break

        # Apply crowd detection
        frame = detect_crowd(frame)

        # Display the frame
        cv2.imshow("Real-Time Crowd Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

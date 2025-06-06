# Eye Detector Module

This module provides functionality for detecting and tracking eye landmarks in images and video streams.

## Requirements
- Python 3.6+
- Install dependencies: `pip install -r requirements.txt`

## Usage Example
```python
from eye_detector import EyeDetector
import cv2

# Initialize detector
detector = EyeDetector()

# Process a single image
image = cv2.imread('your_image.jpg')
eye_landmarks = detector.get_eye_landmarks(image)
detector.draw_landmarks(image, eye_landmarks)

# Or process video
cap = cv2.VideoCapture(0)  # 0 for webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    eye_landmarks = detector.get_eye_landmarks(frame)
    detector.draw_landmarks(frame, eye_landmarks)
    cv2.imshow('Eye Landmarks', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## File Structure
- `eye_detector.py`: Main detector class
- `service/`: Core detection and landmark modules
- `weights/`: Model files
    - RFB-320.tflite: Face detection model
    - sparse_face.tflite: Facial landmark model

## Output Format
The `get_eye_landmarks()` function returns a list of dictionaries, one for each face detected:
- 'right_eye': Points 36-41 (clockwise from outer corner)
- 'left_eye': Points 42-47 (clockwise from outer corner)
Each point is a 2D coordinate (x, y) in the frame's coordinate system.

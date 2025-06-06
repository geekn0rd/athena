#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import service

class EyeDetector:
    def __init__(self, face_detection_model="weights/RFB-320.tflite", 
                 landmark_model="weights/sparse_face.tflite",
                 detection_threshold=0.95):
        """Initialize the eye detector with the required models.
        
        Args:
            face_detection_model (str): Path to the face detection model
            landmark_model (str): Path to the facial landmark model
            detection_threshold (float): Confidence threshold for face detection
        """
        print(f"[Eye Detector] Initializing with detection threshold: {detection_threshold}")
        
        # Initialize face detector
        self.face_detector = service.UltraLightFaceDetecion(
            face_detection_model, 
            conf_threshold=detection_threshold
        )
        
        # Initialize landmark detector
        self.landmark_detector = service.DepthFacialLandmarks(landmark_model)
        print("[Eye Detector] Successfully initialized face and landmark detectors")
        
    def get_eye_landmarks(self, frame):
        """Extract eye landmarks from a frame.
        
        Args:
            frame: Input image/frame (BGR format)
            
        Returns:
            list: List of dictionaries containing eye landmarks for each detected face
                 Each dictionary contains 'left_eye' and 'right_eye' landmarks
        """
        # Detect faces
        boxes, scores = self.face_detector.inference(frame)
        
        if len(boxes) == 0:
            print("[Eye Detector] No faces detected in frame")
            return []
            
        print(f"[Eye Detector] Detected {len(boxes)} faces with confidence scores: {[f'{s:.2f}' for s in scores]}")
        
        # Store results for all faces
        all_eye_landmarks = []
        
        # Get landmarks for each face
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            try:
                # Convert generator to list to make it subscriptable
                results = list(self.landmark_detector.get_landmarks(frame, [box]))
                if not results:
                    print(f"[Eye Detector] No landmarks found for face {idx+1}")
                    continue
                
                landmarks = np.round(results[0][0]).astype(int)  # Access first result's landmarks
                
                # Extract eye landmarks
                right_eye = landmarks[36:42]  # Right eye points (36-41)
                left_eye = landmarks[42:48]   # Left eye points (42-47)
                
                # Validate eye landmarks
                if len(right_eye) != 6 or len(left_eye) != 6:
                    print(f"[Eye Detector] Invalid number of eye landmarks for face {idx+1}")
                    continue
                
                all_eye_landmarks.append({
                    'right_eye': right_eye,
                    'left_eye': left_eye
                })
                print(f"[Eye Detector] Successfully extracted landmarks for face {idx+1}")
                
            except Exception as e:
                print(f"[Eye Detector] Error processing face {idx+1}: {str(e)}")
                continue
            
        return all_eye_landmarks
    
    def draw_landmarks(self, frame, eye_landmarks, color=(224, 255, 255)):
        """Draw eye landmarks on the frame.
        
        Args:
            frame: Input image/frame to draw on
            eye_landmarks: List of eye landmarks from get_eye_landmarks()
            color: BGR color tuple for drawing
        """
        for face_eyes in eye_landmarks:
            # Draw right eye
            for p in face_eyes['right_eye']:
                cv2.circle(frame, tuple(p), 2, color, -1, cv2.LINE_AA)
            cv2.polylines(frame, [face_eyes['right_eye']], True, color, 
                         thickness=1, lineType=cv2.LINE_AA)
            
            # Draw left eye
            for p in face_eyes['left_eye']:
                cv2.circle(frame, tuple(p), 2, color, -1, cv2.LINE_AA)
            cv2.polylines(frame, [face_eyes['left_eye']], True, color, 
                         thickness=1, lineType=cv2.LINE_AA)

# Example usage:
if __name__ == "__main__":
    # Initialize detector
    detector = EyeDetector()
    
    # Open video capture (0 for webcam or video file path)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get eye landmarks
        eye_landmarks = detector.get_eye_landmarks(frame)
        
        # Draw landmarks
        detector.draw_landmarks(frame, eye_landmarks)
        
        # Display result
        cv2.imshow('Eye Landmarks', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 
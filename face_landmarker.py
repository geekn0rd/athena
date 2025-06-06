#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import service

class FaceLandmarker:
    """A class for detecting and visualizing facial landmarks."""
    
    # Define facial feature indices
    FACIAL_FEATURES = {
        'jaw': slice(0, 17),
        'right_eyebrow': slice(17, 22),
        'left_eyebrow': slice(22, 27),
        'nose_bridge': slice(27, 31),
        'nose_tip': slice(31, 36),
        'right_eye': slice(36, 42),
        'left_eye': slice(42, 48),
        'outer_mouth': slice(48, 60),
        'inner_mouth': slice(60, 68)
    }
    
    def __init__(self, face_detection_model="weights/RFB-320.tflite", 
                 landmark_model="weights/sparse_face.tflite",
                 detection_threshold=0.95):
        """Initialize the face detector with the required models.
        
        Args:
            face_detection_model (str): Path to the face detection model
            landmark_model (str): Path to the facial landmark model
            detection_threshold (float): Confidence threshold for face detection
        """
        # Initialize face detector
        self.face_detector = service.UltraLightFaceDetecion(
            face_detection_model, 
            conf_threshold=detection_threshold
        )
        
        # Initialize landmark detector
        self.landmark_detector = service.DepthFacialLandmarks(landmark_model)
        
    def get_landmarks(self, frame):
        """Extract all facial landmarks from a frame.
        
        Args:
            frame: Input image/frame (BGR format)
            
        Returns:
            list: List of dictionaries containing facial landmarks for each detected face
                 Each dictionary contains landmarks for different facial features
        """
        # Detect faces
        boxes, scores = self.face_detector.inference(frame)
        
        # Store results for all faces
        all_face_landmarks = []
        
        # Get landmarks for each face
        for results in self.landmark_detector.get_landmarks(frame, boxes):
            landmarks = np.round(results[0]).astype(int)
            
            # Create a dictionary with all facial features
            face_features = {}
            for feature_name, feature_slice in self.FACIAL_FEATURES.items():
                face_features[feature_name] = landmarks[feature_slice]
            
            # Add additional useful information
            face_features['all_points'] = landmarks  # All 68 points
            face_features['rotation'] = results[1] if len(results) > 1 else None  # Rotation matrix if available
            
            all_face_landmarks.append(face_features)
            
        return all_face_landmarks
    
    def draw_landmarks(self, frame, face_landmarks, color=(224, 255, 255), 
                      features=None, point_size=2, line_thickness=1):
        """Draw facial landmarks on the frame.
        
        Args:
            frame: Input image/frame to draw on
            face_landmarks: List of facial landmarks from get_landmarks()
            color: BGR color tuple for drawing
            features: List of feature names to draw (None for all features)
            point_size: Size of landmark points
            line_thickness: Thickness of connecting lines
        """
        if features is None:
            features = list(self.FACIAL_FEATURES.keys())
            
        for face in face_landmarks:
            # Draw each requested facial feature
            for feature in features:
                points = face[feature]
                
                # Draw points
                for p in points:
                    cv2.circle(frame, tuple(p), point_size, color, -1, cv2.LINE_AA)
                
                # Connect points with lines
                if len(points) > 2:  # Only draw lines if we have more than 2 points
                    # Close the loop for eyes and mouth
                    if feature in ['right_eye', 'left_eye', 'outer_mouth', 'inner_mouth']:
                        cv2.polylines(frame, [points], True, color, 
                                    thickness=line_thickness, lineType=cv2.LINE_AA)
                    else:
                        cv2.polylines(frame, [points], False, color, 
                                    thickness=line_thickness, lineType=cv2.LINE_AA)
    
    def get_face_orientation(self, landmarks):
        """Get face orientation if rotation matrix is available.
        
        Args:
            landmarks: Single face landmarks dictionary from get_landmarks()
            
        Returns:
            dict: Dictionary containing pitch, yaw, and roll angles in degrees,
                 or None if rotation information is not available
        """
        if landmarks['rotation'] is None:
            return None
            
        def rotation_matrix_to_euler_angles(R):
            """Convert rotation matrix to euler angles."""
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            
            if sy < 1e-6:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0
            else:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
                
            return np.degrees([x, y, z])
        
        angles = rotation_matrix_to_euler_angles(landmarks['rotation'])
        return {
            'pitch': angles[0],  # Up/down
            'yaw': angles[1],    # Left/right
            'roll': angles[2]    # Tilt
        }

# Example usage:
if __name__ == "__main__":
    # Initialize detector
    detector = FaceLandmarker()
    
    # Open video capture (0 for webcam or video file path)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get facial landmarks
        face_landmarks = detector.get_landmarks(frame)
        
        # Draw all landmarks
        detector.draw_landmarks(frame, face_landmarks)
        
        # Optional: Draw only specific features
        # detector.draw_landmarks(frame, face_landmarks, 
        #                        features=['right_eye', 'left_eye', 'nose_tip'])
        
        # Optional: Get face orientation
        if face_landmarks:
            orientation = detector.get_face_orientation(face_landmarks[0])
            if orientation:
                # Display orientation angles
                text = f"Pitch: {orientation['pitch']:.1f}, Yaw: {orientation['yaw']:.1f}, Roll: {orientation['roll']:.1f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (224, 255, 255), 2)
        
        # Display result
        cv2.imshow('Facial Landmarks', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 
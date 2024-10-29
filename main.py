import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

class HandGestureRecognizer:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture tracking
        self.prev_gesture = None
        self.gesture_history = deque(maxlen=5)
        self.last_gesture_time = time.time()
        
        # Finger landmarks
        self.THUMB = 4
        self.INDEX = 8
        self.MIDDLE = 12
        self.RING = 16
        self.PINKY = 20

    def check_finger_state(self, landmarks, finger_tip):
        """Check if a finger is raised"""
        tip_y = landmarks.landmark[finger_tip].y
        pip_y = landmarks.landmark[finger_tip - 2].y  # PIP joint
        return tip_y < pip_y

    def check_thumb(self, landmarks):
        """Special check for thumb position"""
        thumb_tip = landmarks.landmark[self.THUMB].x
        thumb_base = landmarks.landmark[self.THUMB - 2].x
        return thumb_tip < thumb_base

    def recognize_gesture(self, landmarks):
        # Check each finger
        thumb_up = self.check_thumb(landmarks)
        index_up = self.check_finger_state(landmarks, self.INDEX)
        middle_up = self.check_finger_state(landmarks, self.MIDDLE)
        ring_up = self.check_finger_state(landmarks, self.RING)
        pinky_up = self.check_finger_state(landmarks, self.PINKY)
        
        # Recognize basic gestures
        if thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
            return "THUMBS UP"
            
        if index_up and middle_up and not any([thumb_up, ring_up, pinky_up]):
            return "PEACE"
            
        if all([thumb_up, index_up, middle_up, ring_up, pinky_up]):
            return "HIGH FIVE"
            
        if not any([thumb_up, index_up, middle_up, ring_up, pinky_up]):
            return "FIST"
            
        if index_up and not any([thumb_up, middle_up, ring_up, pinky_up]):
            return "POINTING"
            
        return "UNKNOWN"

    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            self.mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # Get gesture
            gesture = self.recognize_gesture(hand_landmarks)
            
            # Simple gesture smoothing
            if gesture != "UNKNOWN":
                self.gesture_history.append(gesture)
                if len(self.gesture_history) >= 3:
                    gesture = max(set(self.gesture_history), key=self.gesture_history.count)
            
            # Display gesture
            cv2.putText(
                frame, 
                f"Gesture: {gesture}", 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        return frame

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()
    
    print("Starting gesture recognition... Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Process frame
        frame = recognizer.process_frame(frame)
        
        # Show frame
        cv2.imshow('Hand Gesture Recognition', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
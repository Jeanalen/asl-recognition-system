import cv2
import mediapipe as mp
import numpy as np
import os
import time

def collect_asl_data():
    # Create directories
    os.makedirs("data/training_data", exist_ok=True)
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    current_letter_index = 0
    
    # Collection parameters
    samples_per_letter = 30
    current_samples = 0
    collecting = False
    countdown = 0
    
    while cap.isOpened() and current_letter_index < len(alphabet):
        success, image = cap.read()
        if not success:
            print("Failed to read from camera")
            continue
        
        # Flip the image horizontally
        image = cv2.flip(image, 1)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image to detect hands
        results = hands.process(image_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # If collecting data and we have landmarks
                if collecting and countdown <= 0:
                    # Extract features
                    features = []
                    for landmark in hand_landmarks.landmark:
                        features.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Save features
                    letter = alphabet[current_letter_index]
                    filename = f"data/training_data/{letter}_{current_samples}.npy"
                    np.save(filename, features)
                    
                    current_samples += 1
                    print(f"Collected sample {current_samples}/{samples_per_letter} for letter {letter}")
                    
                    # Move to next letter if we have enough samples
                    if current_samples >= samples_per_letter:
                        current_letter_index += 1
                        current_samples = 0
                        collecting = False
                        print(f"Moving to letter {alphabet[current_letter_index] if current_letter_index < len(alphabet) else 'Done'}")
        
        # Display current letter to show
        current_letter = alphabet[current_letter_index] if current_letter_index < len(alphabet) else "Done"
        cv2.putText(
            image, 
            f"Show sign for: {current_letter}", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            (0, 255, 0), 
            2
        )
        
        # Display collection status
        if collecting:
            if countdown > 0:
                cv2.putText(
                    image, 
                    f"Starting in: {countdown//10}", 
                    (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
                countdown -= 1
            else:
                cv2.putText(
                    image, 
                    f"Collecting: {current_samples}/{samples_per_letter}", 
                    (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
        else:
            cv2.putText(
                image, 
                "Press 'c' to start collecting", 
                (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
        
        # Show the frame
        cv2.imshow('ASL Data Collection', image)
        
        # Process key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not collecting:
            collecting = True
            countdown = 30  # 3 seconds at 10 frames per second
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting ASL Training Data Collection")
    print("Show the ASL sign for each letter when prompted")
    print("Press 'c' to start collecting each letter")
    print("Press 'q' to quit")
    collect_asl_data()
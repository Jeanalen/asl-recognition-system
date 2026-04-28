import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

class ASLRecognizer:
    def __init__(self):
        # mediapipe historically exposed models under ``mp.solutions``;
        # recent (0.10+) releases use the Tasks API instead.  We try the legacy
        # interface first and fall back to the new API if ``mp.solutions`` is
        # missing.  The rest of the code is written to be agnostic to which
        # backend is actually in use.
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.use_tasks_api = False
        except AttributeError:
            # legacy API unavailable; configure the Tasks backend
            try:
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core.base_options import BaseOptions
            except ImportError:
                raise ImportError(
                    "mediapipe installation does not expose a usable API. "
                    "Make sure you have either a <0.10 release or a >=0.10 release "
                    "with the Python Tasks package installed."
                )

            # allow user to specify a model path or download a default bundle
            model_path = os.getenv("HAND_LANDMARKER_MODEL", "hand_landmarker.task")
            if not os.path.exists(model_path):
                url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                )
                try:
                    print(f"downloading hand landmarker model from {url}...")
                    import urllib.request

                    urllib.request.urlretrieve(url, model_path)
                    print(f"model saved to {model_path}")
                except Exception as e:  # network error or permission
                    raise ImportError(
                        "Unable to obtain hand_landmarker.task model. "
                        "Set HAND_LANDMARKER_MODEL to a valid path or download the "
                        "bundle from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models"
                    ) from e

            base_options = BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            self.hands = vision.HandLandmarker.create_from_options(options)

            # drawing utilities look identical but live under the tasks namespace
            self.mp_hands = vision.HandLandmarksConnections
            self.mp_drawing = vision.drawing_utils
            self.mp_drawing_styles = vision.drawing_styles
            self.use_tasks_api = True
        
        self.asl_model = self.load_or_train_model()
        
        self.current_word = []
        self.last_letter = None
        self.letter_started = False
        self.stable_frames = 0
        self.required_stable_frames = 15  
        self.display_text = ""
        self.completed_words = []
        
    def load_or_train_model(self):
        """Load the trained model or create a basic one if none exists"""
        model_file = "models/asl_model.pkl"
        
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                print("Loaded existing ASL recognition model")
                return model
            except:
                print("Failed to load model, creating a new one")

        print("Creating a new ASL recognition model")
        model = KNeighborsClassifier(n_neighbors=5)
        
        X = np.random.rand(26, 21*3)  
        y = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        model.fit(X, y)
        
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    
    def extract_features(self, hand_landmarks):
        """Extract features from hand landmarks for classification"""
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)
    
    def predict_asl_letter(self, hand_landmarks):
        """Predict the ASL letter from hand landmarks"""
        features = self.extract_features(hand_landmarks)

        landmarks = hand_landmarks.landmark

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]

        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        thumb_extended = thumb_tip.y < wrist.y
        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y
        ring_extended = ring_tip.y < ring_mcp.y
        pinky_extended = pinky_tip.y < pinky_mcp.y

        index_curled = index_tip.y > index_pip.y
        middle_curled = middle_tip.y > middle_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y

        thumb_right = thumb_tip.x > index_mcp.x
        thumb_left = thumb_tip.x < pinky_mcp.x

        if (not index_extended and not middle_extended and not ring_extended and not pinky_extended
                and thumb_right):
            return "A"

        elif (index_extended and middle_extended and ring_extended and pinky_extended
                and abs(index_tip.x - middle_tip.x) < 0.05
                and abs(middle_tip.x - ring_tip.x) < 0.05
                and abs(ring_tip.x - pinky_tip.x) < 0.05):
            return "B"

        elif (index_extended and middle_extended and ring_extended and pinky_extended
                and thumb_tip.x < index_tip.x
                and abs(index_tip.y - thumb_tip.y) < 0.1):
            return "C"

        elif (index_extended and not middle_extended and not ring_extended and not pinky_extended):
            return "D"

        elif (index_curled and middle_curled and ring_curled and pinky_curled):
            return "E"

        elif (not index_extended and middle_extended and ring_extended and pinky_extended
                and abs(thumb_tip.x - index_tip.x) < 0.05
                and abs(thumb_tip.y - index_tip.y) < 0.05):
            return "F"

        elif (index_extended and not middle_extended and not ring_extended and not pinky_extended
                and thumb_extended
                and index_tip.x > index_mcp.x):
            return "G"

        elif (index_extended and middle_extended and not ring_extended and not pinky_extended
                and abs(index_tip.y - middle_tip.y) < 0.05):
            return "H"

        elif (not index_extended and not middle_extended and not ring_extended and pinky_extended):
            return "I"

        elif (not index_extended and not middle_extended and not ring_extended and pinky_extended
                and pinky_tip.x > pinky_mcp.x):
            return "J"

        elif (index_extended and middle_extended and not ring_extended and not pinky_extended
                and abs(index_tip.x - middle_tip.x) > 0.08):
            return "K"

        elif (index_extended and not middle_extended and not ring_extended and not pinky_extended
                and thumb_extended
                and abs(thumb_tip.x - index_mcp.x) < 0.05):
            return "L"

        elif (not index_extended and not middle_extended and not ring_extended and not pinky_extended
                and not thumb_extended):
            return "M"

        elif (not index_extended and not middle_extended and ring_extended and not pinky_extended
                and not thumb_extended):
            return "N"

        elif (not index_extended and not middle_extended and not ring_extended and not pinky_extended
                and abs(thumb_tip.x - index_tip.x) < 0.05
                and abs(thumb_tip.y - index_tip.y) < 0.05):
            return "O"

        elif (index_extended and not middle_extended and not ring_extended and not pinky_extended
                and index_tip.y > index_mcp.y):
            return "P"

        elif (index_extended and not middle_extended and not ring_extended and not pinky_extended
                and index_tip.y > index_mcp.y
                and index_tip.x < index_mcp.x):
            return "Q"

        elif (index_extended and middle_extended and not ring_extended and not pinky_extended
                and abs(index_tip.x - middle_tip.x) < 0.03
                and abs(index_tip.y - middle_tip.y) > 0.05):
            return "R"

        elif (not index_extended and not middle_extended and not ring_extended and not pinky_extended
                and thumb_tip.z < index_mcp.z):
            return "S"

        elif (not index_extended and not middle_extended and not ring_extended and not pinky_extended
                and thumb_extended
                and thumb_tip.x > index_mcp.x
                and thumb_tip.x < middle_mcp.x):
            return "T"

        elif (index_extended and middle_extended and not ring_extended and not pinky_extended
                and abs(index_tip.x - middle_tip.x) < 0.04):
            return "U"

        elif (index_extended and middle_extended and not ring_extended and not pinky_extended
                and abs(index_tip.x - middle_tip.x) > 0.06):
            return "V"

        elif (index_extended and middle_extended and ring_extended and not pinky_extended
                and abs(index_tip.x - middle_tip.x) > 0.05
                and abs(middle_tip.x - ring_tip.x) > 0.05):
            return "W"

        elif (not index_extended and not middle_extended and not ring_extended and not pinky_extended
                and index_tip.z < index_pip.z):
            return "X"

        elif (not index_extended and not middle_extended and not ring_extended and pinky_extended
                and thumb_extended):
            return "Y"

        elif (index_extended and not middle_extended and not ring_extended and not pinky_extended
                and index_tip.x < index_mcp.x
                and index_tip.y > index_mcp.y):
            return "Z"

        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "D"  
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "U"  
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "A"
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return "B" 

        import random
        return random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    def complete_word(self):
        """Finalize the current word"""
        if self.current_word:
            word = ''.join(self.current_word)
            self.completed_words.append(word)
            self.display_text = f"Word: {word}"
            self.current_word = []
            self.last_letter = None
            self.stable_frames = 0
            self.letter_started = False
            return True
        return False
    
    def add_to_word(self, letter):
        """Add recognized letter to the current word"""
        if letter != self.last_letter:
            self.last_letter = letter
            self.stable_frames = 1
        else:
            self.stable_frames += 1

        if self.stable_frames >= self.required_stable_frames and not self.letter_started:
            self.letter_started = True
            self.current_word.append(letter)
            self.display_text = f"Current: {''.join(self.current_word)}"
            self.letter_started = False 
            self.stable_frames = 0
        
        return False
    
    def run(self):
        """Run the ASL recognition system"""
        cap = cv2.VideoCapture(0)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from camera")
                continue

            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            word_complete = False
            detected_letter = None

            # run the appropriate pipeline depending on the backend we created
            if self.use_tasks_api:
                # tasks API expects an mp.Image
                mp_image = mp.Image(mp.ImageFormat.SRGB, image_rgb)
                results = self.hands.detect(mp_image)
                hands_list = results.hand_landmarks or []
                # each entry in hands_list is itself a list of landmarks
                multi_hand_landmarks = []
                for hand in hands_list:
                    # wrap into an object with `.landmark` so downstream code works
                    wrapper = type("HandWrapper", (), {"landmark": hand})()
                    multi_hand_landmarks.append((wrapper, hand))
            else:
                results = self.hands.process(image_rgb)
                multi_hand_landmarks = []
                if results.multi_hand_landmarks:
                    for hand in results.multi_hand_landmarks:
                        multi_hand_landmarks.append((hand, hand))

            # iterate over collected landmarks; each tuple is (for_predict, for_draw)
            for pred_landmarks, draw_landmarks in multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    draw_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                detected_letter = self.predict_asl_letter(pred_landmarks)
                self.add_to_word(detected_letter)

            if detected_letter:
                cv2.putText(
                    image, 
                    f"Detected: {detected_letter}", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    (255, 0, 0), 
                    2
                )

            cv2.putText(
                image, 
                self.display_text, 
                (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )

            if self.completed_words:
                joined_words = " ".join(self.completed_words)
                cv2.putText(
                    image, 
                    f"Completed: {joined_words}", 
                    (10, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (128, 0, 128), 
                    2
                )

            key = cv2.waitKey(5) & 0xFF
            if key == ord('k'):
                word_complete = self.complete_word()
            elif key == ord('q'):
                break

            if word_complete:
                overlay = image.copy()
                cv2.rectangle(
                    overlay, 
                    (0, frame_height // 2 - 50), 
                    (frame_width, frame_height // 2 + 50), 
                    (0, 0, 0), 
                    -1
                )

                completed_word = self.completed_words[-1]
                text_size = cv2.getTextSize(
                    completed_word, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2.5, 
                    3
                )[0]
                text_x = (frame_width - text_size[0]) // 2
                cv2.putText(
                    overlay, 
                    completed_word, 
                    (text_x, frame_height // 2 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2.5, 
                    (255, 255, 255), 
                    3
                )

                alpha = 0.7
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                cv2.imshow('ASL Recognition', image)
                cv2.waitKey(1500)

            cv2.putText(
                image, 
                "Make ASL signs A-Z, press 'k' to complete word", 
                (10, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 0), 
                2
            )

            cv2.imshow('ASL Recognition', image)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = ASLRecognizer()
    recognizer.run()
import os
import json

MODEL_PATH = "hand_landmarker.task"
DB_FILE = "custom_gestures.json"

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                data = json.load(f)
                cleaned = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        cleaned[k] = v
                    elif isinstance(v, list):
                        cleaned[k] = {"vector": v, "keys": ["space"]}
                return cleaned
        except Exception:
            return {}
    return {}

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)

def extract_features_for_single_hand(landmarks, prev_landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    features = []
    for lm in landmarks:
        features.append(lm.x - base_x)
        features.append(lm.y - base_y)
        features.append(lm.z - base_z)

    if prev_landmarks:
        for lm, plm in zip(landmarks, prev_landmarks):
            features.append(lm.x - plm.x)
            features.append(lm.y - plm.y)
    else:
        for _ in range(len(landmarks) * 2):
            features.append(0.0)

    return features

def extract_features_two_hands(hand_landmarks_list, prev_hands_list):
    # Supports up to 2 hands. If a hand slot is empty, it pads with zeros.
    features = []
    new_prev_list = [None, None]
    
    for i in range(2):
        if hand_landmarks_list and i < len(hand_landmarks_list):
            lm = hand_landmarks_list[i]
            plm = prev_hands_list[i] if prev_hands_list and i < len(prev_hands_list) else None
            features.extend(extract_features_for_single_hand(lm, plm))
            new_prev_list[i] = lm
        else:
            features.extend([0.0] * (21 * 5))
            new_prev_list[i] = None
            
    return features, new_prev_list

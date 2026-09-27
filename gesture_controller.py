import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import keyboard
from gesture_trainer import load_db, extract_features_two_hands, MODEL_PATH

def normalize_key(key_name):
    # Common shortcut aliases ko automatic sahi key name mein convert karega
    key_name = key_name.strip().lower()
    if key_name in ['prtscr', 'prtscrn', 'snapshot', 'print_screen']:
        return 'print screen'
    return key_name

def run_controller():
    db = load_db()
    if not db:
        print("[WARNING] No gestures found in database! Please run add_new_gesture.py first.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    prev_hands = [None, None]
    
    last_action_time = 0
    cooldown = 1.0  # 1 second gap between triggers

    print("\n[INFO] Gesture Controller (Dual-Hand) is running smoothly! Press 'q' in the video window to exit.")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # FRAME SKIPPING: CPU load aur cursor lag hatane ke liye har 2nd frame process hoga
        if frame_count % 2 != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        display_frame = cv2.flip(frame, 1)

        if detection_result.hand_landmarks:
            vec, prev_hands = extract_features_two_hands(detection_result.hand_landmarks, prev_hands)

            matched_gesture = None
            min_dist = float('inf')
            matched_keys = []

            for name, data in db.items():
                if not isinstance(data, dict):
                    continue
                stored_vec = data.get("vector", [])
                keys = data.get("keys", [])

                if not stored_vec:
                    continue

                dist = sum((a - b) ** 2 for a, b in zip(vec, stored_vec)) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    matched_gesture = name
                    matched_keys = keys

            if min_dist < 0.5 and matched_gesture:
                current_time = time.time()
                if current_time - last_action_time > cooldown:
                    # Normalize keys automatically (e.g., prtscr -> print screen)
                    normalized_keys = [normalize_key(k) for k in matched_keys]
                    print(f"[ACTION] Triggered '{matched_gesture}' -> Keys: {normalized_keys}")
                    try:
                        if len(normalized_keys) == 1:
                            keyboard.send(normalized_keys[0])
                        else:
                            keyboard.send('+'.join(normalized_keys))
                    except Exception as e:
                        print(f"[ERROR] Could not execute shortcut: {e}")
                    last_action_time = current_time

                cv2.putText(display_frame, f"Action: {matched_gesture}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Searching...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            prev_hands = [None, None]
            cv2.putText(display_frame, "No Hand Detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("GestureFlow - Controller", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_controller()

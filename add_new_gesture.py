import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gesture_trainer import load_db, save_db, extract_features_two_hands, MODEL_PATH

def draw_hand_dots(frame, landmarks):
    h, w, _ = frame.shape
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

def add_new_gesture():
    db = load_db()

    print("\n--- GestureFlow Smart Trainer (Dual-Hand Enabled) ---")
    gesture_name = input("Enter Gesture Name (e.g., screenshot): ").strip().lower()
    if not gesture_name:
        print("Invalid name!")
        return

    action_input = input("Enter Shortcut Keys separated by comma (e.g., win,prtscrn): ").strip()
    action_keys = [k.strip().lower() for k in action_input.split(',')]

    print(f"\nGet ready! Camera will open. You have 5 seconds to position your hands for '{gesture_name}'...")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,  # 2 hands enabled
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    # 1. Live Preparation Phase (5 seconds)
    countdown_duration = 5.0
    start_countdown = time.time()
    
    while cap.isOpened():
        elapsed = time.time() - start_countdown
        remaining = max(0.0, countdown_duration - elapsed)
        if remaining == 0:
            break

        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        display_frame = cv2.flip(frame, 1)

        has_hands = bool(detection_result.hand_landmarks)
        if has_hands:
            for hand_lms in detection_result.hand_landmarks:
                draw_hand_dots(display_frame, hand_lms)

        status_text = f"Get Ready: {remaining:.1f}s" if has_hands else f"Show Hand(s)!: {remaining:.1f}s"
        color = (0, 255, 0) if has_hands else (0, 0, 255)

        cv2.putText(display_frame, status_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("GestureFlow - Manager", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # 2. Actual Recording Phase (7 seconds)
    samples = []
    prev_hands = [None, None]
    record_duration = 7.0
    start_recording = time.time()

    print("\nRECORDING STARTED! Perform your gesture smoothly...")

    while cap.isOpened():
        elapsed = time.time() - start_recording
        remaining_time = max(0.0, record_duration - elapsed)
        if elapsed > record_duration:
            break

        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        display_frame = cv2.flip(frame, 1)

        if detection_result.hand_landmarks:
            for hand_lms in detection_result.hand_landmarks:
                draw_hand_dots(display_frame, hand_lms)
            
            vec, prev_hands = extract_features_two_hands(detection_result.hand_landmarks, prev_hands)
            samples.append(vec)

            cv2.putText(display_frame, f"Recording... {remaining_time:.1f}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            prev_hands = [None, None]
            cv2.putText(display_frame, f"NO HAND DETECTED! {remaining_time:.1f}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("GestureFlow - Manager", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) > 5:
        avg_vector = [sum(col) / len(col) for col in zip(*samples)]
        db[gesture_name] = {
            "vector": avg_vector,
            "keys": action_keys
        }
        save_db(db)
        print(f"\n[SUCCESS] Gesture '{gesture_name}' saved successfully with {len(samples)} valid frames!")
    else:
        print("\n[FAILED] Not enough hand data captured. Please try again.")

if __name__ == "__main__":
    add_new_gesture()
    input("\nPress Enter to exit trainer...")

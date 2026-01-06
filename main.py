import cv2
import time
import mediapipe as mp

# ================== Model paths ==================
hand_model = "./hand_landmarker.task"
gesture_model = "./gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

# ================== Global state ==================
latest_landmarks = None
latest_handedness = None
current_gesture = "None"

# ================== HandLandmarker callback ==================
def hand_callback(result: HandLandmarkerResult, output_image, timestamp_ms):
    global latest_landmarks, latest_handedness
    if result.hand_landmarks:
        latest_landmarks = result.hand_landmarks
        latest_handedness = result.handedness
    else:
        latest_landmarks = None
        latest_handedness = None

# ================== GestureRecognizer callback ==================
def gesture_callback(result, output_image, timestamp_ms):
    global current_gesture
    if result.gestures:
        current_gesture = result.gestures[0][0].category_name
    else:
        current_gesture = "None"

# ================== Options ==================
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,   # 👈 Key: detect up to 1 hand
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=hand_callback
)

# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python
gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_model),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_callback
)

# ------------------ Open webcam ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
     GestureRecognizer.create_from_options(gesture_options) as gesture_recognizer:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int(time.time() * 1000)

        # ===== Tasks inference =====
        hand_landmarker.detect_async(mp_image, timestamp_ms)
        gesture_recognizer.recognize_async(mp_image, timestamp_ms)

        # ===== Manually draw skeleton =====
        if latest_landmarks and latest_handedness:
            for idx, hand in enumerate(latest_landmarks):
                hand_label = latest_handedness[idx][0].category_name  # "Left" or "Right"

                # Different colors for left and right hands
                if hand_label == "Left":
                    color = (255, 0, 0)   # Blue
                else:
                    color = (0, 255, 0)   # Green
                #for hand in latest_landmarks:
                # First draw points
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Calculate radius based on z
                    z = lm.z
                    scale = - z * 0.5   # 0.5 is scaling factor
                    radius = int(200 * scale)

                    # Prevent too small or too large
                    radius = max(1, min(radius, 10))

                    cv2.circle(frame, (cx, cy), radius, color, -1)
                    #cv2.circle(frame, (cx, cy), 4, color, -1)

                # ===== Draw solid lines for five fingers =====
                FINGER_CONNECTIONS = [
                    [1, 2, 3, 4],        # Thumb
                    [5, 6, 7, 8],        # Index
                    [9, 10, 11, 12],     # Middle
                    [13, 14, 15, 16],    # Ring
                    [17, 18, 19, 20]     # Pinky
                ]
                # Draw lines (five fingers)
                for finger in FINGER_CONNECTIONS:
                    for i in range(len(finger) - 1):
                        start = hand[finger[i]]
                        end = hand[finger[i + 1]]

                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)

                        z_base = hand[8].z  # Use index fingertip as base
                        scale = -z_base
                        thickness = int(50 * scale)
                        thickness = max(1, min(thickness, 6))
                        
                        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
                # Connect thumb root
                start = hand[1]
                end = hand[0]

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)

                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                # Connect roots of four fingers
                for i in range(5, 14, 4):
                    start = hand[i]
                    end = hand[i+4]

                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)

                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                # ===== Wrist connections =====
                cv2.line(frame,
                 (int(hand[5].x * w), int(hand[5].y * h)),
                 (int(hand[0].x * w), int(hand[0].y * h)),
                 color, 2)

                cv2.line(frame,
                 (int(hand[17].x * w), int(hand[17].y * h)),
                 (int(hand[0].x * w), int(hand[0].y * h)),
                 color, 2)
                # ===== Display Left / Right text =====
                text_x = int(hand[0].x * w)
                text_y = int(hand[0].y * h) - 10
                cv2.putText(frame, hand_label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Use index fingertip as reference and draw a fixed size circle for comparison
                z_base = hand[8].z  # Use index fingertip as base
                x_base = hand[8].x
                y_base = hand[8].y
                
                x_b = int(hand[8].x * w)
                y_b = int(hand[8].y * h)

                cv2.circle(frame, (x_b, y_b), 4, (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"X--: {x_base:.4f}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Y--: {y_base:.4f}",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Z--: {z_base:.4f}",
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )

        # ===== Display gesture =====
        cv2.putText(
            frame,
            f"Gesture: {current_gesture}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Tasks Hand + Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

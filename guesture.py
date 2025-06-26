import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Finger tip landmarks
finger_tips = [4, 8, 12, 16, 20]

# Colors and font
font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (255, 255, 255)
bg_color = (0, 0, 0)

# Open webcam
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, hand_label, w, h):
    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    fingers = []

    # Thumb (depends on hand)
    if hand_label == "Right":
        fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)
    else:
        fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)

    # Other fingers
    for tip_id in finger_tips[1:]:
        fingers.append(1 if lm_list[tip_id][1] < lm_list[tip_id - 2][1] else 0)

    return fingers

def detect_gesture(fingers):
    if fingers == [0, 1, 1, 0, 0]:
        return "Peace ✌️"
    elif fingers == [0, 1, 0, 0, 0]:
        return "Pointing ☝️"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist 👊"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Hand 🖐️"
    else:
        return ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    black_bg = np.zeros_like(frame)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    total_fingers = 0
    gesture_texts = []

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(black_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand label (left or right)
            hand_label = "Right"  # default
            if results.multi_handedness:
                hand_label = results.multi_handedness[i].classification[0].label

            fingers = count_fingers(hand_landmarks, hand_label, w, h)
            total_fingers += sum(fingers)

            gesture = detect_gesture(fingers)
            if gesture:
                gesture_texts.append(f"{hand_label}: {gesture}")

    # Display finger count
    cv2.putText(black_bg, f"Total Fingers: {total_fingers}", (30, 70), font, 1.8, font_color, 4)

    # Show gestures
    for i, text in enumerate(gesture_texts):
        cv2.putText(black_bg, text, (30, 130 + i * 50), font, 1.3, font_color, 3)

    cv2.imshow("Gesture Vision - Premium Mode", black_bg)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()

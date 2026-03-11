
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import math
import time
from collections import deque, Counter
import os

MODEL_PATH = "models/two_stream_best.h5"
CLASSES_FILE = "classes.npy"
CLASSES = ['hello', 'iloveyou', 'thankyou']
NUM_FRAMES = 16
FRAME_SIZE = 112
CAMERA_INDEX = 0 
SCALE = 0.75       # hand crop size factor
UP_SHIFT = 0.20    # upward shift fraction
SMOOTH_K = 5       # smoothing window
PREDICT_EVERY_N = 2


# Load model 
if os.path.exists(CLASSES_FILE):
    try:
        CLASSES = np.load(CLASSES_FILE, allow_pickle=True).tolist()
        print("Loaded classes from", CLASSES_FILE, ":", CLASSES)
    except Exception as e:
        print("Could not load classes.npy:", e, "Using fallback list.")

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Mediapipe setup 
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=1)
hand_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4)

# GLOBAL BODY CROPPER 
def crop_gesture_space(frame, pose_results, size=FRAME_SIZE, pad=0.1):
    """Crop upper-body region around shoulders and face (global stream)."""
    h, w, _ = frame.shape
    if not pose_results or not pose_results.pose_landmarks:
        min_dim = min(h, w)
        yc, xc = h // 2, w // 2
        crop = frame[yc - min_dim // 2:yc + min_dim // 2,
                     xc - min_dim // 2:xc + min_dim // 2]
        return cv2.resize(crop, (size, size)), None

    lm = pose_results.pose_landmarks.landmark
    nose = lm[mp_pose.PoseLandmark.NOSE]
    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    nx, ny = int(nose.x * w), int(nose.y * h)
    lshx, lshy = int(l_sh.x * w), int(l_sh.y * h)
    rshx, rshy = int(r_sh.x * w), int(r_sh.y * h)

    x_min = max(0, min(lshx, rshx) - int(pad * w))
    x_max = min(w, max(lshx, rshx) + int(pad * w))
    y_min = max(0, ny - int(0.4 * h))
    y_max = min(h, max(lshy, rshy) + int(0.6 * h))

    box = (x_min, y_min, x_max, y_max)
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        min_dim = min(h, w)
        crop = frame[(h - min_dim) // 2:(h + min_dim) // 2,
                     (w - min_dim) // 2:(w + min_dim) // 2]

    crop = cv2.resize(crop, (size, size))
    return crop, box

# HAND CROPPER 
def single_hand_crop(frame, pose_results, hands_results,
                     side='right', size=FRAME_SIZE,
                     scale=SCALE, up_shift=UP_SHIFT):
    h, w, _ = frame.shape
    if not pose_results or not pose_results.pose_landmarks:
        min_dim = min(h, w)
        crop = frame[(h - min_dim)//2:(h + min_dim)//2,
                     (w - min_dim)//2:(w + min_dim)//2]
        return cv2.resize(crop, (size, size)), None

    lm = pose_results.pose_landmarks.landmark
    wrist = mp_pose.PoseLandmark.RIGHT_WRIST if side == 'right' else mp_pose.PoseLandmark.LEFT_WRIST
    elbow = mp_pose.PoseLandmark.RIGHT_ELBOW if side == 'right' else mp_pose.PoseLandmark.LEFT_ELBOW

    xw, yw = int(lm[wrist].x * w), int(lm[wrist].y * h)
    xe, ye = int(lm[elbow].x * w), int(lm[elbow].y * h)
    L = int(math.hypot(xw - xe, yw - ye))
    if L < 8:
        L = min(h, w) // 4

    xB = max(0, xw - int(L * scale))
    xE = min(w, xw + int(L * scale))
    yB = max(0, yw - int(L * scale * (1 + up_shift)))
    yE = min(h, yw + int(L * scale * (1 - up_shift)))

    if xE <= xB or yE <= yB:
        min_dim = min(h, w) // 3
        xB = max(0, (w - min_dim)//2)
        xE = min(w, (w + min_dim)//2)
        yB = max(0, (h - min_dim)//2)
        yE = min(h, (h + min_dim)//2)

    crop = frame[yB:yE, xB:xE].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)

    if hands_results and hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            pts = []
            for lm_h in hand_landmarks.landmark:
                px = int(lm_h.x * w) - xB
                py = int(lm_h.y * h) - yB
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            if pts.size > 0:
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
    else:
        ch, cw = crop.shape[:2]
        cv2.circle(mask, (cw//2, ch//2), max(ch, cw)//2, 255, -1)

    seg = cv2.bitwise_and(crop, crop, mask=mask)
    seg = cv2.convertScaleAbs(seg, alpha=1.15, beta=8)
    seg = cv2.resize(seg, (size, size))
    return seg, (xB, yB, xE, yE)

# MERGE BOTH HANDS
def merge_hands(frame, pose_results, hands_results, size=FRAME_SIZE):
    right_crop, _ = single_hand_crop(frame, pose_results, hands_results, side='right', size=size)
    left_crop, _ = single_hand_crop(frame, pose_results, hands_results, side='left', size=size)
    merged = cv2.addWeighted(right_crop, 0.5, left_crop, 0.5, 0)
    return merged

# PREPROCESS CLIP 
def preprocess_clip(global_frames, local_frames):
    Xg = np.array(global_frames, dtype='float32') / 255.0
    Xl = np.array(local_frames, dtype='float32') / 255.0
    return [Xg[np.newaxis, ...], Xl[np.newaxis, ...]]


cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

global_frames = deque(maxlen=NUM_FRAMES)
local_frames = deque(maxlen=NUM_FRAMES)
pred_history = deque(maxlen=SMOOTH_K)
frame_counter = 0

print("Ready — press 'q' to exit.")


cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

global_frames = deque(maxlen=NUM_FRAMES)
local_frames = deque(maxlen=NUM_FRAMES)
pred_history = deque(maxlen=SMOOTH_K)
frame_counter = 0
last_label, last_conf = "Waiting...", 0.0
last_pred_time = 0

print("Ready — press 'q' to exit.")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("Camera read failed, exiting.")
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pose_res = pose_detector.process(frame_rgb)
    hands_res = hand_detector.process(frame_rgb)

    g_crop, g_box = crop_gesture_space(frame_rgb, pose_res, size=FRAME_SIZE, pad=0.08)
    l_crop = merge_hands(frame_rgb, pose_res, hands_res, size=FRAME_SIZE)

    global_frames.append(g_crop)
    local_frames.append(l_crop)

    disp = frame_bgr.copy()
    if g_box:
        x1, y1, x2, y2 = g_box
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
    rcrop, rbox = single_hand_crop(frame_rgb, pose_res, hands_res, side='right')
    lcrop, lbox = single_hand_crop(frame_rgb, pose_res, hands_res, side='left')
    if rbox:
        x1, y1, x2, y2 = rbox
        cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if lbox:
        x1, y1, x2, y2 = lbox
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 165, 255), 2)

    if pose_res.pose_landmarks:
        mp_drawing.draw_landmarks(disp, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if hands_res and hands_res.multi_hand_landmarks:
        for h in hands_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(disp, h, mp_hands.HAND_CONNECTIONS)

    # show debug crops
    cv2.imshow("Model Global Crop", cv2.cvtColor(g_crop, cv2.COLOR_RGB2BGR))
    cv2.imshow("Model Local Crop (merged hands)", cv2.cvtColor(l_crop, cv2.COLOR_RGB2BGR))

    # detect hand presence
    hands_present = hands_res and hands_res.multi_hand_landmarks and len(hands_res.multi_hand_landmarks) > 0

    frame_counter += 1
    now = time.time()

    # Predict every N frames 
    if hands_present and len(global_frames) == NUM_FRAMES and frame_counter % PREDICT_EVERY_N == 0:
        Xg, Xl = preprocess_clip(list(global_frames), list(local_frames))
        preds = model.predict([Xg, Xl], verbose=0)[0]
        prob_dict = {c: float(round(p, 3)) for c, p in zip(CLASSES, preds)}
        print("probs:", prob_dict)

        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = CLASSES[idx]

        if SMOOTH_K > 0:
            pred_history.append(label)
            label = Counter(pred_history).most_common(1)[0][0]

        last_label, last_conf = label, conf
        last_pred_time = now

    # If no hand, show "no hand" 
    if not hands_present and now - last_pred_time > 2:
        last_label, last_conf = "No hand detected", 0.0

    label_text = f"{last_label} ({last_conf:.2f})" if last_conf > 0 else last_label
    cv2.putText(disp, label_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Sign Detection", disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")

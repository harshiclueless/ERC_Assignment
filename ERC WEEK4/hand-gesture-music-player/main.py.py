import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import pygame

pygame.mixer.init()
cap = cv2.VideoCapture(0)
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,                      
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)
HAND_CONNECTIONS = {(0,1), (1,2), (2,3), (3,4),
                    (0,5), (5,6), (6,7), (7,8),
                    (9,10), (10,11), (11,12),
                    (13,14), (14,15), (15,16),
                    (0,17), (17,18), (18,19), (19,20),
                    (5,9), (9,13), (13,17)
                    }

sounds_mapping = {
    1: "sounds_swaras/1_Sa.mp3",
    2: "sounds_swaras/2_Re.mp3",
    3: "sounds_swaras/3_Ga.mp3",
    4: "sounds_swaras/4_Ma.mp3",
    5: "sounds_swaras/5_Pa.mp3",
    6: "sounds_swaras/6_Dha.mp3",
    7: "sounds_swaras/7_Ni.mp3",
    8: "sounds_swaras/8_Sa_upper.mp3"
}

def draw_landmarks(frame, hand_landmarks):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0,255,0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0,0,255), -1)

start_time = time.time()
previous_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print("Skipping")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    count=0
    
    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            draw_landmarks(frame, hand_landmarks)
            if hand_landmarks[8].y < hand_landmarks[6].y:
              count += 1

            if hand_landmarks[12].y < hand_landmarks[10].y:
              count += 1

            if hand_landmarks[16].y < hand_landmarks[14].y:
              count += 1
    
            if hand_landmarks[20].y < hand_landmarks[18].y:
              count += 1

            label = result.handedness[i][0].category_name 

            if label == "Left":
                if hand_landmarks[4].x < hand_landmarks[3].x:
                    count +=1
            if label == "Right":
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    count +=1

    if count != previous_count and count in sounds_mapping:
     pygame.mixer.music.load(sounds_mapping[count])
     pygame.mixer.music.play()
     previous_count=count
    
    cv2.putText(frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
landmarker.close()
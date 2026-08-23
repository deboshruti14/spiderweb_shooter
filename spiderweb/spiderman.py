import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os

# 1. Initialize Pygame Audio with standard sample rate
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
pygame.mixer.init()

sound_path = "thwip.mp3"
thwip_sound = None
if os.path.exists(sound_path):
    try:
        thwip_sound = pygame.mixer.Sound(sound_path)
    except Exception as e:
        print(f"Audio loading error: {e}")
else:
    print(f"Warning: '{sound_path}' not found in folder.")

# 2. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 3. Load & Auto-Fix Image Transparency
def load_transparent_image(filepath):
    raw_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        raise FileNotFoundError(f"Error: '{filepath}' not found.")
    
    # If the image has no alpha channel (RGB only), convert black/dark or white pixels to transparent
    if raw_img.ndim == 2: # Grayscale
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGRA)
    elif raw_img.shape[2] == 3:
        # Check if background is mostly black or white
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        # Create alpha mask from luminance (bright parts stay visible, dark parts go transparent)
        alpha = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
        b, g, r = cv2.split(raw_img)
        raw_img = cv2.merge([b, g, r, alpha])
    
    return raw_img

splat_img = load_transparent_image('web_splat.png')

# 4. Seamless Alpha Overlay Helper
def overlay_transparent(background, overlay, center_x, center_y, opacity=1.0):
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, _ = overlay.shape

    x1 = int(center_x - ov_w // 2)
    y1 = int(center_y - ov_h // 2)
    x2 = x1 + ov_w
    y2 = y1 + ov_h

    if x1 >= bg_w or y1 >= bg_h or x2 <= 0 or y2 <= 0:
        return background

    clip_x1 = max(0, x1)
    clip_y1 = max(0, y1)
    clip_x2 = min(bg_w, x2)
    clip_y2 = min(bg_h, y2)

    overlay_crop = overlay[clip_y1 - y1 : clip_y2 - y1, clip_x1 - x1 : clip_x2 - x1]
    bg_crop = background[clip_y1:clip_y2, clip_x1:clip_x2]

    alpha = ((overlay_crop[:, :, 3] / 255.0) * opacity)[:, :, np.newaxis]
    rgb = overlay_crop[:, :, :3]

    blended = (alpha * rgb + (1.0 - alpha) * bg_crop).astype(np.uint8)
    background[clip_y1:clip_y2, clip_x1:clip_x2] = blended
    return background

# 5. Gesture Detection (🤘)
def is_spiderman_gesture(lm):
    index_up = lm[8].y < lm[6].y
    pinky_up = lm[20].y < lm[18].y
    middle_down = lm[12].y > lm[10].y
    ring_down = lm[16].y > lm[14].y
    return index_up and pinky_up and middle_down and ring_down

# 6. Web Blast Projectile
class WebBlast:
    def __init__(self, start_x, start_y, target_x, target_y):
        self.start_x = start_x
        self.start_y = start_y
        self.target_x = target_x
        self.target_y = target_y
        self.progress = 0.0
        self.speed = 0.15
        self.state = "FLYING"
        self.splat_opacity = 1.0

    def update(self):
        if self.state == "FLYING":
            self.progress += self.speed
            if self.progress >= 1.0:
                self.progress = 1.0
                self.state = "SPLAT"
        elif self.state == "SPLAT":
            self.splat_opacity -= 0.02

    def is_done(self):
        return self.state == "SPLAT" and self.splat_opacity <= 0

# 7. Main Loop
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

window_name = "Spider-Man 3D Lens Splat AR"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

active_webs = []
last_gesture_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    screen_center_x, screen_center_y = w // 2, h // 2

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            if is_spiderman_gesture(lm):
                wrist_x, wrist_y = int(lm[0].x * w), int(lm[0].y * h)
                
                if current_time - last_gesture_time > 0.6:
                    if thwip_sound:
                        thwip_sound.play()
                    active_webs.append(WebBlast(wrist_x, wrist_y, screen_center_x, screen_center_y))
                    last_gesture_time = current_time

    # Render Active Webs
    for web in active_webs[:]:
        web.update()

        if web.state == "FLYING":
            curr_x = int(web.start_x + (web.target_x - web.start_x) * web.progress)
            curr_y = int(web.start_y + (web.target_y - web.start_y) * web.progress)
            
            current_size = int(80 + (w * 0.9) * (web.progress ** 2))
            
            # Glowing Web Cable
            cv2.line(frame, (web.start_x, web.start_y), (curr_x, curr_y), (200, 200, 200), int(3 + 8 * web.progress))
            cv2.line(frame, (web.start_x, web.start_y), (curr_x, curr_y), (255, 255, 255), 2)

            resized_blast = cv2.resize(splat_img, (current_size, current_size))
            frame = overlay_transparent(frame, resized_blast, curr_x, curr_y, opacity=0.95)

        elif web.state == "SPLAT":
            full_screen_splat = cv2.resize(splat_img, (w, h))
            frame = overlay_transparent(frame, full_screen_splat, screen_center_x, screen_center_y, opacity=web.splat_opacity)

        if web.is_done():
            active_webs.remove(web)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
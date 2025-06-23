from re import X
import cv2
import numpy as np
import time
import random

# === CONFIGURATION ===
VIDEO_PATH = "fire_video.mp4"   # Path to your fire video
GRID_SIZE = 20                  # Grid size (20x20)
SPREAD_PROB = 0.3               # Fire spread probability
SPREAD_INTERVAL = 10            # Spread every 10 frames (slow spread)
DELAY = 200                     # 200 ms between frames (slow motion)

# === VIDEO SETUP ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cell_w = frame_width // GRID_SIZE
cell_h = frame_height // GRID_SIZE

# === FIRE GRID ===
fire_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

frame_count = 0
prev_centroid = None
start_time = time.time()

# === CREATE WINDOW & TRACKBAR BEFORE LOOP ===
window_name = "🔥 Drone Fire + Spread Direction"
cv2.namedWindow(window_name)
cv2.createTrackbar("Threshold", window_name, 240, 255, lambda x: None)

# === MAIN LOOP ===
while cap.isOpened():
    if time.time() - start_time > 60:
        print("⏱️ 1 minute reached.")
        break

    ret, frame = cap.read()
    if not ret:
        break

    # 🔁 Get threshold from trackbar
    THRESHOLD = cv2.getTrackbarPos("Threshold", window_name)

    # === FIRE DETECTION ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        gx = cx // cell_w
        gy = cy // cell_h
        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            fire_grid[gy, gx] = 1  # burning

    # === FIRE SPREAD SIMULATION ===
    if frame_count % SPREAD_INTERVAL == 0:
        new_grid = fire_grid.copy()
        for i in range(1, GRID_SIZE - 1):
            for j in range(1, GRID_SIZE - 1):
                if fire_grid[i, j] == 1:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            ni, nj = i + dy, j + dx
                            if fire_grid[ni, nj] == 0 and random.random() < SPREAD_PROB:
                                new_grid[ni, nj] = 1
                    new_grid[i, j] = 2  # mark as burned
        fire_grid = new_grid

    # === DIRECTION DETECTION ===
    burning_cells = np.argwhere(fire_grid == 1)
    if len(burning_cells) > 0:
        centroid = np.mean(burning_cells, axis=0).astype(int)
        if prev_centroid is not None:
            dx = centroid[1] - prev_centroid[1]
            dy = centroid[0] - prev_centroid[0]
            direction = ""
            if abs(dx) > 1:
                direction += "Right " if dx > 0 else "Left "
            if abs(dy) > 1:
                direction += "Down" if dy > 0 else "Up"
            if direction.strip() == "":
                direction = "Stable"

            cv2.putText(frame, f"Spread: {direction.strip()}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            pt1 = (prev_centroid[1] * cell_w + cell_w // 2, prev_centroid[0] * cell_h + cell_h // 2)
            pt2 = (centroid[1] * cell_w + cell_w // 2, centroid[0] * cell_h + cell_h // 2)
            cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 3)

        prev_centroid = centroid

    # === DRAW FIRE GRID ===
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            top_left = (j * cell_w, i * cell_h)
            bottom_right = ((j + 1) * cell_w, (i + 1) * cell_h)
            if fire_grid[i, j] == 1:
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)
            elif fire_grid[i, j] == 2:
                cv2.rectangle(frame, top_left, bottom_right, (50, 50, 50), -1)

    # === DRAW GRID LINES ===
    for i in range(1, GRID_SIZE):
        cv2.line(frame, (0, i * cell_h), (frame_width, i * cell_h), (80, 80, 80), 1)
    for j in range(1, GRID_SIZE):
        cv2.line(frame, (j * cell_w, 0), (j * cell_w, frame_height), (80, 80, 80), 1)

    # === DISPLAY FRAME ===
    cv2.imshow(window_name, frame)
    if cv2.waitKey(DELAY) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

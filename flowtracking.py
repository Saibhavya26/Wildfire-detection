import cv2
import numpy as np
import time
import random

# === CONFIGURATION ===
VIDEO_PATH = "fire_video.mp4"
GRID_SIZE = 20
SPREAD_PROB = 0.3
SPREAD_INTERVAL = 10     # 🔁 Spread every 10 frames (slower spread)
THRESHOLD = 180
DELAY = 150              # ⏱️ Delay in ms between frames (slower animation)

# === VIDEO SETUP ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cell_w = frame_width // GRID_SIZE
cell_h = frame_height // GRID_SIZE

fire_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
source_map = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

frame_count = 0
prev_centroid = None
start_time = time.time()

def update_thresh(x):
    global THRESHOLD
    THRESHOLD = x

cv2.namedWindow("🔥 Drone Fire + Spread Direction")
cv2.createTrackbar("Threshold", "🔥 Drone Fire + Spread Direction", THRESHOLD, 255, update_thresh)

while cap.isOpened():
    if time.time() - start_time > 120:  # Run for 2 minutes
        print("⏱️ 2 minutes reached.")
        break

    ret, frame = cap.read()
    if not ret:
        break

    THRESHOLD = cv2.getTrackbarPos("Threshold", "🔥 Drone Fire + Spread Direction")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        gx = cx // cell_w
        gy = cy // cell_h
        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            fire_grid[gy, gx] = 1

    # === FIRE SPREAD ===
    if frame_count % SPREAD_INTERVAL == 0:
        new_grid = fire_grid.copy()
        new_source_map = [row.copy() for row in source_map]

        for i in range(1, GRID_SIZE - 1):
            for j in range(1, GRID_SIZE - 1):
                if fire_grid[i, j] == 1:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            ni, nj = i + dy, j + dx
                            if fire_grid[ni, nj] == 0 and random.random() < SPREAD_PROB:
                                new_grid[ni][nj] = 1
                                new_source_map[ni][nj] = (i, j)
                    new_grid[i][j] = 2

        fire_grid = new_grid
        source_map = new_source_map

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

    # === DRAW FLOW ARROWS ===
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if source_map[i][j] is not None:
                src_i, src_j = source_map[i][j]
                pt1 = (src_j * cell_w + cell_w // 2, src_i * cell_h + cell_h // 2)
                pt2 = (j * cell_w + cell_w // 2, i * cell_h + cell_h // 2)
                cv2.arrowedLine(frame, pt1, pt2, (255, 0, 0), 1)

    # === DRAW GRID ===
    for i in range(1, GRID_SIZE):
        cv2.line(frame, (0, i * cell_h), (frame_width, i * cell_h), (80, 80, 80), 1)
    for j in range(1, GRID_SIZE):
        cv2.line(frame, (j * cell_w, 0), (j * cell_w, frame_height), (80, 80, 80), 1)

    # === SHOW FRAME ===
    cv2.imshow("🔥 Drone Fire + Spread Direction", frame)
    if cv2.waitKey(DELAY) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

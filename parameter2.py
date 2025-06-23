import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
video_path = "fire_video3.mp4"
cap = cv2.VideoCapture(video_path)

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0
prev_fire_area = 0
prev_centroid = (0, 0)

# Fixed values
threshold_value = 180
fire_cell_value = 1
unburnt_cell_value = 0
burnt_cell_value = 2
wind_speed = 15
humidity = 30
spread_direction = "East"

# Tracking lists for graph
time_list = []
fire_cells_list = []
smoke_cells_list = []
spread_rate_list = []
confidence_list = []

print("🎥 Displaying video. Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # Loop the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        prev_fire_area = 0
        prev_centroid = (0, 0)
        continue

    time_elapsed = frame_idx / frame_rate

    # FIRE MASK
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fire_lower = np.array([10, 100, 100])
    fire_upper = np.array([30, 255, 255])
    fire_mask = cv2.inRange(hsv, fire_lower, fire_upper)
    fire_area = np.count_nonzero(fire_mask)

    # SMOKE MASK
    smoke_lower = np.array([0, 0, 150])
    smoke_upper = np.array([180, 50, 255])
    smoke_mask = cv2.inRange(hsv, smoke_lower, smoke_upper)
    smoke_area = np.count_nonzero(smoke_mask)

    # Brightness
    brightness = np.mean(hsv[:, :, 2][fire_mask > 0]) if fire_area > 0 else 0

    # Spread Rate
    spread_rate = (fire_area - prev_fire_area) * frame_rate if frame_idx > 0 else 0
    prev_fire_area = fire_area

    # Confidence
    confidence = min(1.0, brightness / 255)

    # Centroid
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            if cx > prev_centroid[0]:
                spread_direction = "East"
            elif cx < prev_centroid[0]:
                spread_direction = "West"
            elif cy < prev_centroid[1]:
                spread_direction = "North"
            elif cy > prev_centroid[1]:
                spread_direction = "South"
            prev_centroid = (cx, cy)
        cv2.drawContours(frame, [largest], -1, (0, 0, 255), 2)

    # Overlay Panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30
    step = 25
    white = (255, 255, 255)
    yellow = (0, 255, 255)

    cv2.putText(frame, "🔥🔥 PARAMETER PANEL", (10, y), font, 0.7, yellow, 2); y += step
    cv2.putText(frame, f"Threshold Value: {threshold_value}", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"Fire Cell Value: {fire_cell_value}", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"Unburnt Cell: {unburnt_cell_value}", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"Burnt Cell: {burnt_cell_value}", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"Spread Direction: {spread_direction}", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"Wind Speed: {wind_speed} m/s", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"Humidity: {humidity}%", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"Simulation Time: {int(time_elapsed)}s", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"🔥 Fire Cells: {fire_area}", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"💨 Smoke Cells: {smoke_area}", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"📈 Spread Rate: {int(spread_rate)}", (10, y), font, 0.6, white, 1); y += step
    cv2.putText(frame, f"✅ Confidence: {confidence:.2f}", (10, y), font, 0.6, white, 1); y += step

    # Show frame
    cv2.imshow("🔥 Fire Parameter Panel", frame)
    frame_idx += 1

    # Track values
    time_list.append(time_elapsed)
    fire_cells_list.append(fire_area)
    smoke_cells_list.append(smoke_area)
    spread_rate_list.append(spread_rate)
    confidence_list.append(confidence)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === AFTER VIDEO: Show Graphs ===
plt.figure(figsize=(14, 10))

# 🔥 Fire Cells
plt.subplot(2, 2, 1)
plt.plot(time_list, fire_cells_list, color='red')
plt.title("🔥 Fire Cells Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Number of Fire Cells")
plt.grid(True)

# 💨 Smoke Cells
plt.subplot(2, 2, 2)
plt.plot(time_list, smoke_cells_list, color='gray')
plt.title("💨 Smoke Cells Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Number of Smoke Cells")
plt.grid(True)

# 📈 Spread Rate
plt.subplot(2, 2, 3)
plt.plot(time_list, spread_rate_list, color='blue')
plt.title("📈 Spread Rate Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Spread Rate")
plt.grid(True)

# ✅ Confidence
plt.subplot(2, 2, 4)
plt.plot(time_list, confidence_list, color='green')
plt.title("✅ Confidence Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Confidence Score")
plt.grid(True)

plt.tight_layout()
plt.show()
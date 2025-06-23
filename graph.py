import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
video_path = "fire_video3.mp4"  # Make sure the video is in the same directory or update the path

# === Open Video ===
cap = cv2.VideoCapture(video_path)

# === Parameters ===
fire_areas = []
avg_brightness = []
centroid_positions = []
frame_times = []

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire color range (tune if needed)
    lower_fire = np.array([10, 100, 100])
    upper_fire = np.array([30, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Fire area
    fire_area = np.count_nonzero(fire_mask)
    fire_areas.append(fire_area)

    # Brightness (V channel average where fire exists)
    v_channel = hsv[:, :, 2]
    brightness = np.mean(v_channel[fire_mask > 0]) if fire_area > 0 else 0
    avg_brightness.append(brightness)

    # Centroid of the largest fire region
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid_positions.append((cx, cy))
        else:
            centroid_positions.append((0, 0))
    else:
        centroid_positions.append((0, 0))

    # Time in seconds
    frame_times.append(frame_idx / frame_rate)
    frame_idx += 1

cap.release()

# === Data for plots ===
centroid_x = [pt[0] for pt in centroid_positions]
centroid_y = [pt[1] for pt in centroid_positions]
spread_speed = np.diff(fire_areas) / np.diff(frame_times)
spread_speed = np.insert(spread_speed, 0, 0)  # Match lengths

# === Plot Graphs ===
plt.figure(figsize=(14, 10))

# 1. Fire Area
plt.subplot(2, 2, 1)
plt.plot(frame_times, fire_areas, color='orange')
plt.title("🔥 Fire Area Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Fire Area (pixels)")
plt.grid(True)

# 2. Brightness
plt.subplot(2, 2, 2)
plt.plot(frame_times, avg_brightness, color='red')
plt.title("💡 Average Brightness Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Brightness (V-channel)")
plt.grid(True)

# 3. Centroid
plt.subplot(2, 2, 3)
plt.plot(frame_times, centroid_x, label="Centroid X", color='blue')
plt.plot(frame_times, centroid_y, label="Centroid Y", color='green')
plt.title("📍 Centroid Position Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend()
plt.grid(True)

# 4. Spread Speed
plt.subplot(2, 2, 4)
plt.plot(frame_times, spread_speed, color='purple')
plt.title("⚡ Fire Spread Speed Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Spread Speed (pixels/sec)")
plt.grid(True)

plt.tight_layout()
plt.savefig("fire_video_analysis_graphs.png")  # Saves the graph image
plt.show()

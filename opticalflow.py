import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configuration
GRID_SIZE = 100
CELL_SIZE = 5
SPREAD_PROB = 0.3
STEPS = 30

# States: 0 = unburned, 1 = burning, 2 = burned
fire_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
fire_grid[GRID_SIZE // 2, GRID_SIZE // 2] = 1

frames = []

def update_fire(grid):
    new_grid = grid.copy()
    for i in range(1, GRID_SIZE - 1):
        for j in range(1, GRID_SIZE - 1):
            if grid[i, j] == 1:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        ni, nj = i + dy, j + dx
                        if grid[ni, nj] == 0 and np.random.rand() < SPREAD_PROB:
                            new_grid[ni, nj] = 1
                new_grid[i, j] = 2
    return new_grid

def render_grid(grid):
    vis = np.ones((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE, 3), dtype=np.uint8) * 255
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = (255, 255, 255)
            if grid[i, j] == 1:
                color = (0, 0, 255)
            elif grid[i, j] == 2:
                color = (50, 50, 50)
            cv2.rectangle(vis,
                          (j * CELL_SIZE, i * CELL_SIZE),
                          ((j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE),
                          color, -1)
    return vis

# Generate simulation frames
for _ in range(STEPS):
    frame = render_grid(fire_grid)
    frames.append(frame.copy())
    fire_grid = update_fire(fire_grid)

# Compute optical flow and overlay arrows
output_frames = []
for i in range(1, len(frames)):
    prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    vis = frames[i].copy()
    step = 10
    for y in range(0, vis.shape[0], step):
        for x in range(0, vis.shape[1], step):
            dx, dy = flow[y, x]
            mag = np.sqrt(dx*2 + dy*2)
            if mag > 1:
                color = (0, int(min(255, mag*10)), 255 - int(min(255, mag*10)))
                cv2.arrowedLine(vis, (x, y), (int(x + dx), int(y + dy)), color, 2, tipLength=0.4)
    output_frames.append(vis)

# Animate
fig, ax = plt.subplots()
im = ax.imshow(output_frames[0])
ax.set_title("🔥 SimFire Optical Flow (Clear Arrows)")
ax.axis("off")

def update_plot(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update_plot, frames=output_frames,
                              interval=1000, blit=True)

plt.show()
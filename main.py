# main.py
# Autonomous Robot Navigation Simulator (without pygame)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from planning.grid_map import GridMap
from planning.astar import astar
from robot.pid import PIDController

CELL_SIZE = 1

class Robot:
    def __init__(self, path):
        self.path = path
        self.index = 0

        self.x = path[0][0]
        self.y = path[0][1]

        self.pid_x = PIDController(0.2, 0.01, 0.05)
        self.pid_y = PIDController(0.2, 0.01, 0.05)

    def update(self):
        if self.index >= len(self.path):
            return

        target = self.path[self.index]

        vx = self.pid_x.compute(target[0], self.x)
        vy = self.pid_y.compute(target[1], self.y)

        self.x += vx * 0.1
        self.y += vy * 0.1

        if abs(self.x - target[0]) < 0.2 and \
           abs(self.y - target[1]) < 0.2:
            self.index += 1


grid = GridMap(40, 30)

start = (2, 2)
goal = (35, 25)

path = astar(grid, start, goal)

robot = Robot(path)

fig, ax = plt.subplots(figsize=(10, 7))

def animate(frame):
    ax.clear()

    # Draw obstacles
    for obstacle in grid.obstacles:
        ax.add_patch(
            plt.Rectangle(
                obstacle,
                1,
                1
            )
        )

    # Draw path
    px = [p[0] for p in path]
    py = [p[1] for p in path]

    ax.plot(px, py)

    # Update robot
    robot.update()

    # Draw robot
    ax.plot(robot.x, robot.y, marker='o', markersize=12)

    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)

    ax.set_title("Autonomous Robot Navigation")

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=500,
    interval=50
)

plt.show()
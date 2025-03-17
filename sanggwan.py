import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

height = [155.3,157.5,156.5,163.9,169.3,170.5,195.5,175.1,173.8,177.7,182.6,180.7,186.7,189.9,189.2]
foot_size = [220,230,235,240,245,250,255,260,265,270,275,280,285,290,300]

foot_size = np.concatenate((foot_size[:6], foot_size[7:]))
height = np.concatenate((height[:6], height[7:]))

xm = np.mean(foot_size)
ym = np.mean(height)

a = np.sum((foot_size - xm) * (height - ym)) / np.sum((foot_size - xm) ** 2)
b = ym - a * xm

y_pred = a * foot_size +b
ss_total = np.sum((height - ym) ** 2)
ss_residual = np.sum((height - y_pred) ** 2)
r_squared = 1 - (ss_residual / ss_total)

line_x = np.linspace(min(foot_size), max(foot_size), 100)
line_y = a * line_x + b

angle = math.degrees(math.atan(a))

plt.figure(figsize=(8,6))


plt.plot(line_x, line_y, 'r--', label="Regression Line")
plt.scatter(foot_size, height,c='g', s=50, alpha=0.7, edgecolors='purple',label = "Actual Data")

for x in range(len(foot_size)):
    plt.text(foot_size[x] + 2, height[x] + 0.5, f"({foot_size[x]}, {height[x]})", fontsize=8)

plt.text(min(foot_size), max(height)-5, f"angle: {angle:.2f}", fontsize=10, color = "blue")
plt.text(min(foot_size), max(height)-10, f"R^2:: {r_squared:.3f}", fontsize=10, color = "blue")

plt.xlabel("Foot Size (mm)")
plt.ylabel("Height (cm)")
plt.legend()
plt.grid(True)
plt.show()
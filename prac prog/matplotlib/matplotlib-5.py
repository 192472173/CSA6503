import matplotlib.pyplot as plt

height = [150, 155, 160, 165, 170, 175]
weight = [45, 50, 55, 60, 65, 70]

plt.scatter(height, weight, color="red")
plt.title("Height vs Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")

plt.show()
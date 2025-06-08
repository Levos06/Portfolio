import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from noise import snoise4

# Настройки
scale = 10.0
octaves = 6
height = 4.0
size_x, size_y = 200, 100
frames = size_y + size_x

# Сетка
x = np.linspace(0, scale, size_x)
y = np.linspace(0, scale, size_y)
x, y = np.meshgrid(x, y)

# Семена шума
seed_z = np.random.uniform(0, 100)
seed_w = np.random.uniform(0, 100)

# Векторизированный шум
def generate_noise(x, y, seed, offset):
    return snoise4(x / scale, y / scale, seed, offset, octaves=octaves)

vec_noise = np.vectorize(generate_noise)
z = vec_noise(x, y, seed_z, 0.0) * height
w = vec_noise(x, y, seed_w, 1.0) * height
points4D = np.stack([x, y, z, w], axis=-1)

# Проекционные векторы
a = np.random.randn(4)
a /= np.linalg.norm(a)
b = np.random.randn(4)
b -= np.dot(b, a) * a
b /= np.linalg.norm(b)

# Проекции
u = np.tensordot(points4D, a, axes=([-1], [0]))
v = np.tensordot(points4D, b, axes=([-1], [0]))

# Подготовка фигуры
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor("white")
ax.set_aspect('equal')
ax.axis('off')

# Установка пределов, чтобы не скакало
margin = 0.1
ax.set_xlim(np.min(u) - margin, np.max(u) + margin)
ax.set_ylim(np.min(v) - margin, np.max(v) + margin)

# Все возможные линии
h_lines = [ax.plot([], [], color='black', alpha=0.2, linewidth=0.2)[0] for _ in range(size_y)]
v_lines = [ax.plot([], [], color='black', alpha=0.2, linewidth=0.2)[0] for _ in range(size_x)]
all_lines = h_lines + v_lines

def update(frame):
    for i in range(frame + 1):  # кумулятивное добавление
        if i < size_y:
            h_lines[i].set_data(u[i, :], v[i, :])
        else:
            j = i - size_y
            v_lines[j].set_data(u[:, j], v[:, j])
    return all_lines[:frame+1]

ani = FuncAnimation(fig, update, frames=frames, interval=10, blit=True)
plt.tight_layout()
plt.show()

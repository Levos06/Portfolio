import numpy as np
import matplotlib.pyplot as plt
from noise import snoise4
from time import time

# Настройки
scale = 10.0
octaves = 6
height = 4.0
n_projections = 10

# Основной таймер
t0 = time()


# Фигура
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor("white")


for _ in range(n_projections):
    size_x = int(np.random.uniform(200, 250))
    size_y = int(np.random.uniform(125, 200))
    # Сетка (2D)
    x = np.linspace(0, scale, size_x)
    y = np.linspace(0, scale, size_y)
    x, y = np.meshgrid(x, y)
    # Генерация нового 4D-многообразия
    z = np.zeros_like(x)
    w = np.zeros_like(x)
    seed_z = np.random.uniform(0, 100)
    seed_w = np.random.uniform(0, 100)
    for i in range(size_y):
        for j in range(size_x):
            z[i, j] = snoise4(x[i, j] / scale, y[i, j] / scale,
                              seed_z, 0.0, octaves=octaves) * height
            w[i, j] = snoise4(x[i, j] / scale, y[i, j] / scale,
                              seed_w, 1.0, octaves=octaves) * height

    points4D = np.stack([x, y, z, w], axis=-1)

    # Случайные ортонормированные векторы для проекции
    a = np.random.randn(4)
    a /= np.linalg.norm(a)
    b = np.random.randn(4)
    b -= np.dot(b, a) * a
    b /= np.linalg.norm(b)


    # Проекции
    u = np.tensordot(points4D, a, axes=([-1], [0]))
    v = np.tensordot(points4D, b, axes=([-1], [0]))

    # Рисуем сетку
    for i in range(size_y):
        ax.plot(u[i, :], v[i, :], color='black', alpha=0.16, linewidth=0.3)
    for j in range(size_x):
        ax.plot(u[:, j], v[:, j], color='black', alpha=0.14, linewidth=0.3)
# Настройка вида
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
print("Время:", time() - t0)
plt.savefig("/Users/levosadchi/Desktop/Новая папка", dpi=500)
plt.show()


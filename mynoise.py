import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from noise import snoise4

# Параметры
scale = 0.03
octaves = 6
height = 0.02
size_x, size_y = 150, 100
frames = 1000
n_projections = 2  # количество проекций

# Сетка (2D)
x = np.linspace(0, scale, size_x)
y = np.linspace(0, scale, size_y)
x, y = np.meshgrid(x, y)

# Семена
seed_z = np.random.uniform(0, 100)
seed_w = np.random.uniform(0, 100)

# Создаем фигуру с подграфиками
ncols = int(np.ceil(np.sqrt(n_projections)))
nrows = int(np.ceil(n_projections / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(7, 7), squeeze=False)
fig.subplots_adjust(wspace=0, hspace=0)

# Настройки осей и проекционные векторы
projections = []
for idx in range(n_projections):
    ax = axes[idx // ncols][idx % ncols]
    ax.axis("off")
    ax.set_aspect('equal')
    a = np.random.randn(4)
    a /= np.linalg.norm(a)
    b = np.random.randn(4)
    b -= np.dot(b, a) * a
    b /= np.linalg.norm(b)
    lines = [ax.plot([], [], color='black', alpha=0.2, linewidth=0.2)[0] for _ in range(size_y + size_x)]
    projections.append((ax, a, b, lines))


# Основная функция обновления
def update(frame):
    t = frame * 0.01

    # Векторизированная генерация z и w
    def generate_component(seed, offset):
        return np.vectorize(lambda i, j: snoise4(x[i, j] / scale,
                                                 y[i, j] / scale,
                                                 seed,
                                                 offset,
                                                 octaves=octaves))(np.indices(x.shape)[0], np.indices(x.shape)[1]) * height

    z = generate_component(seed_z, t)
    w = generate_component(seed_w, t)
    points4D = np.stack([x, y, z, w], axis=-1)

    # Обновление каждой проекции
    for ax, a, b, lines in projections:
        u = np.tensordot(points4D, a, axes=([-1], [0]))
        v = np.tensordot(points4D, b, axes=([-1], [0]))

        for i in range(size_y):
            lines[i].set_data(u[i, :], v[i, :])
        for j in range(size_x):
            lines[size_y + j].set_data(u[:, j], v[:, j])

    return [line for _, _, _, lines in projections for line in lines]


ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
plt.show()

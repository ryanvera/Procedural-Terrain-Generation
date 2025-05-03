import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
import os

def generate_wrapped_diamond_square(size: int, roughness: float, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    grid = np.zeros((size, size), dtype=float)
    init_val = np.random.rand()
    grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = init_val

    step_size = size - 1
    scale = roughness

    while step_size > 1:
        half_step = step_size // 2

        # Diamond step
        for y in range(half_step, size - 1, step_size):
            for x in range(half_step, size - 1, step_size):
                avg = (grid[(y - half_step) % (size - 1), (x - half_step) % (size - 1)] +
                       grid[(y - half_step) % (size - 1), (x + half_step) % (size - 1)] +
                       grid[(y + half_step) % (size - 1), (x - half_step) % (size - 1)] +
                       grid[(y + half_step) % (size - 1), (x + half_step) % (size - 1)]) / 4.0
                grid[y, x] = avg + (np.random.rand() * 2 - 1) * scale

        # Square step
        for y in range(0, size - 1, half_step):
            for x in range((y + half_step) % step_size, size - 1, step_size):
                avg = (grid[y, (x - half_step) % (size - 1)] +
                       grid[y, (x + half_step) % (size - 1)] +
                       grid[(y - half_step) % (size - 1), x] +
                       grid[(y + half_step) % (size - 1), x]) / 4.0
                grid[y, x] = avg + (np.random.rand() * 2 - 1) * scale

        step_size //= 2
        scale *= roughness

    grid[-1, :] = grid[0, :]
    grid[:, -1] = grid[:, 0]
    return grid

def generate_original_diamond_square(size: int, roughness: float, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    grid = np.zeros((size, size), dtype=float)
    grid[0, 0] = np.random.rand()
    grid[0, -1] = np.random.rand()
    grid[-1, 0] = np.random.rand()
    grid[-1, -1] = np.random.rand()

    step_size = size - 1
    scale = roughness

    while step_size > 1:
        half_step = step_size // 2

        for y in range(half_step, size - 1, step_size):
            for x in range(half_step, size - 1, step_size):
                avg = (grid[y - half_step, x - half_step] +
                       grid[y - half_step, x + half_step] +
                       grid[y + half_step, x - half_step] +
                       grid[y + half_step, x + half_step]) / 4.0
                grid[y, x] = avg + (np.random.rand() * 2 - 1) * scale

        for y in range(0, size, half_step):
            for x in range((y + half_step) % step_size, size, step_size):
                s = []
                if x - half_step >= 0: s.append(grid[y, x - half_step])
                if x + half_step < size: s.append(grid[y, x + half_step])
                if y - half_step >= 0: s.append(grid[y - half_step, x])
                if y + half_step < size: s.append(grid[y + half_step, x])
                avg = sum(s) / len(s)
                grid[y, x] = avg + (np.random.rand() * 2 - 1) * scale

        step_size //= 2
        scale *= roughness

    return grid

def apply_radial_falloff(grid: np.ndarray, falloff_strength: float = 3.5) -> np.ndarray:
    h, w = grid.shape
    cx, cy = w / 2, h / 2
    max_dist = np.sqrt(cx**2 + cy**2)
    falloff = np.zeros_like(grid)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            falloff[y, x] = 1 - (dist / max_dist) ** falloff_strength
    return grid * falloff

def normalize_range(grid, min_val=0.0, max_val=1.0):
    return ((grid - np.min(grid)) / (np.max(grid) - np.min(grid))) * (max_val - min_val) + min_val

def smooth(grid, sigma):
    return gaussian_filter(grid, sigma=sigma)

def ensure_output_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def plot(grid, size, title, prefix, folder, show):
    land_type_boundaries = [0, 0.05, 0.20, 0.30, 0.45, 0.70, 0.90, 1]
    land_type_colors = ['darkblue', "blue", "lightblue", 'darkgreen', 'green','grey', 'white']
    cmap = ListedColormap(land_type_colors)
    norm = BoundaryNorm(land_type_boundaries, cmap.N)

    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"{folder}/{prefix}_2d.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    x, y = np.meshgrid(np.linspace(0, size-1, size), np.linspace(0, size-1, size))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=120)
    surf = ax.plot_surface(x, y, np.fliplr(grid), cmap=cmap, norm=norm)
    plt.title(title)
    plt.savefig(f"{folder}/{prefix}_3d.png", dpi=300)
    if show:
        plt.show()
    plt.close()


def run_versions(label, gen_fn, size=1025, roughness=0.5, min_z=0.0, max_z=1.0, show=False):
    folder = f"images/diamond_square_{label}"
    ensure_output_folder(folder)
    base = gen_fn(size, roughness, seed=2024)
    norm_base = normalize_range(base, min_z, max_z)

    plot(norm_base, size, f"{label.title()} Diamond-Square", f"{label}_base", folder, show)

    for sigma in [0.5, 1, 2, 4, 8, 16]:
        smoothed = normalize_range(smooth(norm_base, sigma), min_z, max_z)
        plot(smoothed, size, f"{label.title()} Diamond-Square (σ={sigma})", f"{label}_sigma{sigma}", folder, show)


def run_versions_with_falloff(size=1025, roughness=0.5, min_z=0.0, max_z=1.0, show=False):
    folder = "images/diamond_square_falloff"
    ensure_output_folder(folder)
    base = generate_original_diamond_square(size, roughness, seed=36)
    base = normalize_range(base, min_z, max_z)
    plot(base, size, "Before Falloff", "falloff_before", folder, show)

    island = normalize_range(apply_radial_falloff(base, falloff_strength=3.5), min_z, max_z)
    plot(island, size, "Falloff Island", "falloff", folder, show)



if __name__ == "__main__":
    run_versions("wrapped", generate_wrapped_diamond_square)
    run_versions("original", generate_original_diamond_square)
    run_versions_with_falloff()
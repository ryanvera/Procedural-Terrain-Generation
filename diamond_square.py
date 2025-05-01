import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
import os

def generate_diamond_square(size: int, roughness: float, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    assert (size - 1) & (size - 2) == 0, "Size must be 2^n + 1"
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

def normalize_range(grid: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    min_grid = np.min(grid)
    max_grid = np.max(grid)
    normalized = (grid - min_grid) / (max_grid - min_grid)
    return normalized * (max_val - min_val) + min_val

def apply_radial_falloff(grid: np.ndarray, falloff_strength: float = 3.0) -> np.ndarray:
    h, w = grid.shape
    cx, cy = w / 2, h / 2
    max_dist = np.sqrt(cx**2 + cy**2)

    falloff = np.zeros_like(grid)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            falloff[y, x] = 1 - (dist / max_dist) ** falloff_strength

    return grid * falloff

def smooth_terrain(grid: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return gaussian_filter(grid, sigma=sigma)

def ensure_output_folder():
    folder = "images/diamond_square"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

def plot_terrain(grid: np.ndarray, size: int, title: str, filename_prefix: str, show_plots: bool = True, apply_biomes: bool = True):
    if apply_biomes:
        colors = ['darkblue', "blue", "lightblue", 'darkgreen', 'green','grey', 'white']
        bounds = [0, 0.05, 0.20, 0.30, 0.45, 0.70, 0.90, 1]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
    else:
        cmap = 'terrain'
        norm = None

    plt.imshow(grid, cmap=cmap, norm=norm, interpolation='lanczos')
    if apply_biomes:
        plt.colorbar(ticks=bounds)
    else:
        plt.colorbar()
    plt.title(title)
    plt.savefig(f"images/diamond_square/{filename_prefix}_2d.png", dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    x = np.linspace(0, size - 1, size)
    y = np.linspace(0, size - 1, size)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, grid, cmap=cmap, norm=norm, edgecolor='none')
    if apply_biomes:
        fig.colorbar(surf, ax=ax, boundaries=bounds, ticks=bounds, label='Elevation')
    else:
        fig.colorbar(surf, ax=ax, label='Elevation')
    ax.set_title(title)
    plt.savefig(f"images/diamond_square/{filename_prefix}_3d.png", dpi=300)
    if show_plots:
        plt.show()
    plt.close()

def run_diamond_square_with_falloff(size: int, roughness: float, min_z: float, max_z: float, show_plots: bool = False):
    print("Generating Diamond-Square terrain with radial falloff...")
    terrain = generate_diamond_square(size, roughness, seed=36)
    terrain = normalize_range(terrain, min_val=min_z, max_val=max_z)
    raw = terrain.copy()
    terrain = apply_radial_falloff(terrain, falloff_strength=3.5)
    terrain = normalize_range(terrain, min_val=min_z, max_val=max_z)

    ensure_output_folder()
    plot_terrain(raw, size, "Raw Diamond-Square Before Falloff", "diamond_square_falloff_before", show_plots, apply_biomes=False)
    plot_terrain(terrain, size, "Raw Diamond-Square Island Terrain", "diamond_square_falloff_raw", show_plots, apply_biomes=False)
    plot_terrain(terrain, size, "Biome-Colored Diamond-Square Island Terrain", "diamond_square_falloff_biome", show_plots, apply_biomes=True)

def run_diamond_square_multi_smoothing(size: int, roughness: float, min_z: float, max_z: float, sigmas: list[float], show_plots: bool = False):
    print("Generating multi-sigma smoothed Diamond-Square terrains...")
    base = generate_diamond_square(size, roughness, seed=123)
    base = normalize_range(base, min_val=min_z, max_val=max_z)

    ensure_output_folder()

    for sigma in sigmas:
        print(f"\tApplying Gaussian smoothing with sigma={sigma}...")
        smoothed = smooth_terrain(base, sigma)
        smoothed = normalize_range(smoothed, min_val=min_z, max_val=max_z)

        biome_name = f"diamond_square_sigma{sigma}_biome"
        plot_terrain(smoothed, size, f"Biome Smoothed Diamond-Square (σ={sigma})", biome_name, show_plots, apply_biomes=True)

if __name__ == "__main__":
    run_diamond_square_with_falloff(
        size=1025,
        roughness=0.5,
        min_z=0.0,
        max_z=1.0,
        show_plots=False
    )

    run_diamond_square_multi_smoothing(
        size=513,
        roughness=0.5,
        min_z=0.0,
        max_z=1.0,
        sigmas=[0, 0.5, 1, 2, 4, 8, 16, 32],
        show_plots=False
    )

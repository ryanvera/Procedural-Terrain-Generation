import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def generate_two_tier_voronoi_biomes(size: int, num_major_seeds: int, num_minor_seeds: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    # Define biome types and colors
    biome_types = [
        'desert', 'plains', 'frozen', 'beach', 'forest', 'mountain', 'swamp'
    ]
    biome_colors = [
        '#EDC9AF', '#7CFC00', '#E0FFFF', '#FAFAD2', '#228B22', '#A9A9A9', '#556B2F'
    ]

    num_biomes = len(biome_types)

    # Generate major biome seeds and ensure all biomes are represented
    major_seeds = np.random.randint(0, size, (num_major_seeds, 2))
    major_biome_ids = np.zeros(num_major_seeds, dtype=int)
    major_biome_ids[:num_biomes] = np.arange(num_biomes)
    major_biome_ids[num_biomes:] = np.random.choice(num_biomes, num_major_seeds - num_biomes)

    # Generate minor seeds and assign blended biome colors
    minor_seeds = np.random.randint(0, size, (num_minor_seeds, 2))
    minor_colors = np.zeros((num_minor_seeds, 3))

    for i, seed_point in enumerate(minor_seeds):
        dists = np.sum((major_seeds - seed_point) ** 2, axis=1)
        nearest_two = np.argsort(dists)[:2]
        d1, d2 = dists[nearest_two[0]], dists[nearest_two[1]]
        biome1, biome2 = major_biome_ids[nearest_two[0]], major_biome_ids[nearest_two[1]]

        if d2 < 2 * d1:
            # Corrected Blend colors
            t = d1 / (d1 + d2)
            color1 = np.array(hex_to_rgb(biome_colors[biome1]))
            color2 = np.array(hex_to_rgb(biome_colors[biome2]))
            blended = (1 - t) * color1 + t * color2
            minor_colors[i] = blended / 255.0
        else:
            minor_colors[i] = np.array(hex_to_rgb(biome_colors[biome1])) / 255.0

    # Build the RGB image
    biome_map = np.zeros((size, size, 3))
    for y in range(size):
        for x in range(size):
            dists = np.sum((minor_seeds - np.array([x, y])) ** 2, axis=1)
            nearest = np.argmin(dists)
            biome_map[y, x] = minor_colors[nearest]

    return biome_map, major_seeds

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def ensure_output_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def plot_voronoi(biome_map, seeds, size, title, prefix, folder, show):
    plt.imshow(biome_map)
    plt.scatter(seeds[:, 0], seeds[:, 1], c='black', marker='x', s=20)  # major seeds only
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{folder}/{prefix}_2d.png", dpi=300)
    if show:
        plt.show()
    plt.close()

def run_two_tier_voronoi(size=1000, num_major_seeds=10, num_minor_seeds=800, seed=42, show=False):
    folder = "images/two_tier_voronoi"
    ensure_output_folder(folder)
    biome_map, seeds = generate_two_tier_voronoi_biomes(
        size, num_major_seeds, num_minor_seeds, seed
    )
    plot_voronoi(biome_map, seeds, size, "Two-Tier Voronoi Biome Map (Blended)", "two_tier_voronoi", folder, show)

if __name__ == "__main__":
    run_two_tier_voronoi()

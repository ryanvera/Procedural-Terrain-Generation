import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from noise import pnoise2
import os

def generate_perlin_noise(size, scale=100, octaves=6, persistence=0.5, lacunarity=2.0):
    """
    Generate a 2D Perlin noise array normalized to the range [0, 1].

    :param width: Width of the noise map.
    :param height: Height of the noise map.
    :param scale: Scale of the noise (higher values zoom out).
    :param octaves: Number of levels of detail.
    :param persistence: Amplitude of each octave.
    :param lacunarity: Frequency of each octave.
    :return: 2D numpy array of normalized Perlin noise values.
    """
    width, height = size
    noise_array = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            noise_array[y][x] = pnoise2(x / scale, 
                                        y / scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=width, 
                                        repeaty=height, 
                                        base=42)
            
    # Normalize the noise values to the range [0, 1]
    min_val = noise_array.min()
    max_val = noise_array.max()
    noise_array = (noise_array - min_val) / (max_val - min_val)
    return noise_array

def display_noise(noise_array, output_dir, size, land_type_boundaries, land_type_colors, scale, octaves, persistence, lacunarity):
    """
    Save the Perlin noise as a 2D image or a 3D surface plot.

    :param noise_array: 2D numpy array of Perlin noise values.
    :param mode: Display mode, either "2D" or "3D".
    :param output_dir: Directory to save the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define custom colormap
    cmap = ListedColormap(land_type_colors)
    norm = BoundaryNorm(land_type_boundaries, cmap.N)

    # Create a 2D image
    plt.imshow(noise, cmap=cmap, norm=norm, interpolation='lanczos')
    plt.colorbar()
    plt.title("2D True Perlin Noise")
    plt.text(10, size-75, f"Octaves: {octaves}\nPersistence: {persistence}\nLacunarity: {lacunarity}\nScale: {scale}", fontsize=8, color='red', ha='left', va='top')
    output_path = os.path.join(output_dir, "true_perlin_noise_2D.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


    # Create a 3D surface plot
    height, width = noise_array.shape
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    land = ax.plot_surface(x, y, noise_array, cmap=cmap, norm=norm, edgecolor='none')
    cbar = fig.colorbar(land, ax=ax, boundaries=land_type_boundaries, ticks=land_type_boundaries)
    cbar.set_label('Noise Value')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Perlin Noise Value')
    ax.set_title("3D True Perlin Noise")
    fig.text(0.20, 0.80, f"Octaves: {octaves}\nPersistence: {persistence}\nLacunarity: {lacunarity}\nScale: {scale}", fontsize=8, color='red', ha='left', va='top')
    output_path = os.path.join(output_dir, "true_perlin_noise_3D.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Parameters for the Perlin noise
    size = (512, 512)
    octaves = 1
    persistence = 0.50         # Amplitude of each octave
    lacunarity = 2.0          # Frequency of each octave
    scale = 100               # Scale of the noise (higher values zoom out)

    # Hyperparameters to dictate the land type colors and boundaries
    water_level = 0.30                                                      # Water level for fractal Perlin noise generation
    land_type_boundaries = [0,   0.05,      0.20,   water_level,    0.45,       0.70,   0.90,   1]      # Boundaries for different land types
    land_type_colors = [        'darkblue', "blue", "lightblue", 'darkgreen', 'green','grey', 'white']  # Colors for different land types 



    # Generate Perlin noise
    noise = generate_perlin_noise(size, scale, octaves, persistence, lacunarity)


    # Save Perlin noise in 2D
    display_noise(noise, \
                output_dir="images/true_perlin", 
                land_type_boundaries=land_type_boundaries, \
                land_type_colors=land_type_colors, \
                scale=scale, \
                octaves=octaves, \
                persistence=persistence, \
                lacunarity=lacunarity, \
                size = size[0] \
                )



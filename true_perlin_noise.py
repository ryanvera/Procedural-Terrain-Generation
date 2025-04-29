import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from mpl_toolkits.mplot3d import Axes3D

def generate_perlin_noise(width, height, scale=100, octaves=6, persistence=0.5, lacunarity=2.0):
    """
    Generate a 2D Perlin noise array.

    :param width: Width of the noise map.
    :param height: Height of the noise map.
    :param scale: Scale of the noise (higher values zoom out).
    :param octaves: Number of levels of detail.
    :param persistence: Amplitude of each octave.
    :param lacunarity: Frequency of each octave.
    :return: 2D numpy array of Perlin noise values.
    """
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
    return noise_array

def display_noise(noise_array, mode="2D"):
    """
    Display the Perlin noise as a 2D image or a 3D surface plot.

    :param noise_array: 2D numpy array of Perlin noise values.
    :param mode: Display mode, either "2D" or "3D".
    """
    if mode == "2D":
        plt.imshow(noise_array, cmap='gray')
        plt.colorbar()
        plt.title("2D Perlin Noise")
        plt.show()
    elif mode == "3D":
        height, width = noise_array.shape
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        x, y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, noise_array, cmap='viridis', edgecolor='none')
        ax.set_title("3D Perlin Noise")
        plt.show()
    else:
        raise ValueError("Invalid mode. Use '2D' or '3D'.")

if __name__ == "__main__":
    # Parameters for the Perlin noise
    width = 256
    height = 256
    scale = 100
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    # Generate Perlin noise
    noise = generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity)

    # Display Perlin noise in 2D
    display_noise(noise, mode="2D")

    # Display Perlin noise in 3D
    display_noise(noise, mode="3D")

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import time
import random



def generate_permutation() -> np.ndarray[int]:
    """
    Generates a permutation array for use in Perlin noise generation.
    The function creates an array of integers from 0 to 255, shuffles it randomly,
    and then duplicates the shuffled array to produce a final array of size 512.
    This ensures seamless wrapping when accessing the permutation array.

    Returns:
        np.ndarray[int]: A 1D NumPy array of integers with a length of 512, 
        containing the duplicated and shuffled permutation values.
    """
    # Create a list of integers from 0 to 255
    permutation = np.arange(256, dtype=int)
    np.random.shuffle(permutation)
    return np.tile(permutation, 2) # Duplicate the permutation array to create a size of 512



def fade(t:float) -> float:
    """Computes the fade function as defined by Ken Perlin, which is used to smooth 
    coordinate values in Perlin noise generation. The function transitions values 
    smoothly towards integers, ensuring a gradual and visually appealing interpolation.
    The fade function is defined as:
        f(t) = 6t^5 - 15t^4 + 10t^3
    Args:
        t (float): The input value, typically in the range [0, 1].
    Returns:
        float: The smoothed output value after applying the fade function."""
    # 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)



def grad(hash:int, x:float, y:float=0, z:float=0) -> float:
    """Compute the gradient based on the hash value and coordinates.
    The gradient is computed using a simple hash function that determines the 
    direction of the gradient vector. This function performs a bitwise operation on the hash value
    to determine the gradient direction and then computes the dot product with the input coordinates.

    Args:
        hash (int): The hash value used to determine the gradient direction.
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        z (float): The z-coordinate. Defaults to 0.

    Returns:
        float: The computed gradient value."""
        
    h:int = hash & 15  # Get the last 4 bits of the hash value

    u:float = x if h < 8 else y  # Choose x or y based on the hash value
    v:float = y if h < 4 else (x if h == 12 or h == 14 else z)  # Choose y or z based on the hash value

    # Use the last 2 bits of the hash value to determine the sign of the gradient and return their addtion
    return (u if (h&1) == 0 else -u) + (v if (h&2) == 0 else -v)



def interpolate(a:float, b:float, t:float) -> float:
    """Interpolate between two values using linear interpolation.
    This function smoothly transitions between two values based on the input parameter t.

    Args:
        a (float): The first value to interpolate from.
        b (float): The second value to interpolate to.
        t (float): The interpolation factor, typically in the range [0, 1].

    Returns:
        float: The interpolated value."""
    return a + t * (b - a)



def noise(x:float, y:float, permutations) -> float:
    """
    Generates Perlin noise value for the given 3D coordinates (x, y, z).
    Perlin noise is a gradient noise function that produces smooth, natural-looking
    variations. This implementation computes the noise value by interpolating
    gradients at the corners of a unit cube surrounding the input point.
    
    Args:
        x (float): The x-coordinate of the input point.
        y (float): The y-coordinate of the input point.

    Returns:
        float: A normalized Perlin noise value in the range [0, 1].
    
    Notes:
        - The function assumes a precomputed permutation table `self.permutations`
          for hashing coordinates.
        - The `fade` function is used to smooth the interpolation.
        - The `grad` function computes the gradient at a given corner.
        - The `interplate` function performs linear interpolation between values.
    """
    X = int(np.floor(x)) & 255
    Y = int(np.floor(y)) & 255
        
    # Find relative x, y of point in square
    x -= np.floor(x)
    y -= np.floor(y)
        
    # Compute fade curves for x and y
    u = fade(x)
    v = fade(y)
        
    # Hash coordinates of the 4 corners of the square
    A = permutations[X] + Y
    B = permutations[X + 1] + Y
    AA = permutations[A]
    AB = permutations[A + 1]
    BA = permutations[B]
    BB = permutations[B + 1]
    
    # Calculate gradients at the four corners
    g1 = grad(AA, x, y)
    g2 = grad(BA, x - 1, y)
    g3 = grad(AB, x, y - 1)
    g4 = grad(BB, x - 1, y - 1)
        
    # Interpolate the noise values
    x1 = interpolate(g1, g2, u)
    x2 = interpolate(g3, g4, u)
        
    # Final interpolation
    return interpolate(x1, x2, v)



def generate_perlin_noise(width:int, height:int, scale:float=1.0) -> np.ndarray[float]:
    """
    Generate a 2D array of Perlin noise values. This is a non-smoothed version of Perlin noise.

    Args:
        width (int): The width of the noise array.
        height (int): The height of the noise array.
        scale (float, optional): The scale factor for the noise, which determines the frequency of the noise pattern. Defaults to 1.0.

    Returns:
        np.ndarray[float]: A 2D array of Perlin noise values, where each value represents the noise intensity at a specific point.
    
    Notes:
        The nested for loops iterate over each pixel in the 2D grid (height x width) and compute the Perlin noise value 
        for that pixel using the scaled coordinates and the permutation array.
    """
    
    perms = generate_permutation()  # Generate the permutation array
    noise_grid = np.zeros((height, width), dtype=float)


    for y in range(height):
        for x in range(width):
            noise_grid[y, x] = noise(x * scale, y * scale, perms)  # Generate noise value for each pixel
    return noise_grid



def octave_noise(x, y, permutations, octaves=1, persistence=0.5, lacunarity=2.0, amplitude=1.0) -> float:
    """
    Generate Perlin noise with multiple octaves for more natural patterns.
    This function combines multiple layers of Perlin noise, each with increasing
    frequency and decreasing amplitude, to create more complex and natural-looking
    patterns. The parameters allow control over the number of layers (octaves),
    how much each layer contributes (persistence), and how the frequency changes
    between layers (lacunarity).

    Args:
        x (float): The x-coordinate at which to generate noise.
        y (float): The y-coordinate at which to generate noise.
        permutations (list[int]): A permutation table used for generating noise.
        octaves (int, optional): The number of noise layers to combine. Defaults to 1.
        persistence (float, optional): The factor by which the amplitude decreases 
            for each successive octave. Defaults to 0.5.
        lacunarity (float, optional): The factor by which the frequency increases 
            for each successive octave. Defaults to 2.0.
        amplitude (float, optional): The initial amplitude of the noise. Defaults to 1.0.

    Returns:
        float: The combined noise value, normalized to approximately the range [-1, 1].
    """
    total = 0
    frequency = 1
    max_value = 0  # Used for normalizing result
    
    for octave in range(octaves):
        total += noise(x * frequency, y * frequency, permutations) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
        
    # Normalize the result
    return total / max_value



def generate_fractal_map(width, height, scale=0.1, octaves=4, persistence=0.5, lacunarity=2.0, amplitude=1.0, offset_x=0.0, offset_y=0.0) -> np.ndarray[float]:
    """
    Generates a 2D fractal noise map using Perlin noise and fractal smoothing.
    Args:
        width (int): The width of the noise map in pixels.
        height (int): The height of the noise map in pixels.
        scale (float, optional): The scale factor for the noise. Smaller values zoom out, 
            while larger values zoom in. Defaults to 0.1.
        octaves (int, optional): The number of layers of noise to combine. Higher values 
            add more detail. Defaults to 4.
        persistence (float, optional): The amplitude multiplier for each successive octave. 
            Lower values reduce the contribution of higher octaves. Defaults to 0.5.
        lacunarity (float, optional): The frequency multiplier for each successive octave. 
            Higher values increase the frequency of higher octaves. Defaults to 2.0.
        amplitude (float, optional): The base amplitude of the noise. Defaults to 1.0.
    Returns:
        np.ndarray[float]: A 2D array representing the generated fractal noise map.
    """

    permutations = generate_permutation()  # Generate the permutation array
    noise_map = np.zeros((height, width), dtype=float)
    
    # Offsets can be used to sample different areas of the noise
    # offset_x = random.random() * 100
    # offset_y = random.random() * 100
    
    for y in range(height):
        for x in range(width):
            # Scale coordinates and generate noise
            nx = (x + offset_x) / scale             # Changing the "*" to "/" will invert the scale (larger number equals zoomed in)
            ny = (y + offset_y) / scale             # Changing the "*" to "/" will invert the scale (larger number equals zoomed in)
            noise_map[y][x] = octave_noise(nx, ny, permutations, octaves, persistence, lacunarity, amplitude)
            
    return noise_map



def gaussian_smooth(noise_grid:np.ndarray[float], sigma:float) -> np.ndarray[float]:
    """
    Applies Gaussian smoothing to a 2D noise grid.
    Args:
        noise_grid (np.ndarray[float]): The input 2D noise grid to be smoothed.
        sigma (float): The standard deviation for the Gaussian kernel.
    Returns:
        np.ndarray[float]: The smoothed 2D noise grid.
    """
    return gaussian_filter(noise_grid, sigma=sigma)



def normalize_range(noise_grid:np.ndarray[float], min_val:float=0, max_val:float=1) -> np.ndarray[float]:
    """
    Normalize the values in a 2D noise grid to a specified range.
    Args:
        noise_grid (np.ndarray[float]): The input 2D noise grid to be normalized.
        min_val (float): The minimum value of the normalized range. Defaults to -1.
        max_val (float): The maximum value of the normalized range. Defaults to 1.
    Returns:
        np.ndarray[float]: The normalized 2D noise grid with values in the specified range.
    """
    
    min_noise = np.min(noise_grid)
    max_noise = np.max(noise_grid)

    # Normalize the noise grid to the range [0, 1]
    normalized_grid = (noise_grid - min_noise) / (max_noise - min_noise)
    return normalized_grid



def run_perlin_noise(size:tuple, show_plots:bool) -> None:
    """
    Generates and visualizes 2D Perlin noise as both a 2D image and a 3D surface plot.

    Args:
        size (tuple): A tuple containing the width and height of the Perlin noise grid.
        show_plots (bool): Whether to display the generated plots.

    Returns:
        None

    This function performs the following steps:
    1. Generates a 2D grid of Perlin noise.
    2. Normalizes the noise values to the specified range [min_z, max_z].
    3. Creates a 2D grayscale plot of the noise and saves it as an image.
    4. Creates a 3D surface plot of the noise and saves it as an image.
    5. Displays both the 2D and 3D plots.

    Note:
        - The Perlin noise generation relies on the `generate_perlin_noise` function.
        - The normalization is handled by the `normalize_range` function.
        - The plots are saved in the "images" directory with filenames 
          "perlin_v2_2d.png" and "perlin_v2_3d.png".
    """
    
    #  Create a 2D grid of Perlin noise
    print("Generating Perlin noise...")
    noise = generate_perlin_noise(size[0], size[1], scale=0.1)  # Generate Perlin noise

    # Normalize the noise values to a specified range
    noise = normalize_range(noise)  # Normalize the noise values

    # 2D Plot the generated noise
    plt.imshow(noise, cmap='Greens', interpolation='lanczos')
    plt.colorbar()  # Add a color bar to the plot
    plt.title("Perlin Noise")
    plt.savefig("images/perlin/perlin_2d.png", dpi=300)  # Save the plot as an image
    print("Plotting 2D Perlin noise...")
    if show_plots:
        plt.show()  # Display the plot
    plt.close()
    

    # 3D Plot the generated noise
    x = np.linspace(0, size[0]-1, size[0])
    y = np.linspace(0, size[1]-1, size[1])
    x, y = np.meshgrid(x, y)

    # Plotting the 2D Perlin noise in a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(x, y, noise, cmap='viridis')

    # Labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Perlin Noise Value')
    ax.set_title('Perlin Noise')
    plt.savefig("images/perlin/perlin_3d.png", dpi=300)  # Save the plot as an image
    print("Plotting 3D Perlin noise...\n")
    if show_plots:
        plt.show()  # Display the plot
    plt.close()

    print("Finished generating Perlin noise\n\n")



def run_perlin_noise_fractal(size:tuple, octaves:int, persistence:float, amplitude:float, lacunarity:float, scale:float, colors:list[str], bounds:list[float], show_plots:bool, iterNum:int=0, offset_x=0, offset_y=0) -> None:
    """
    Generates and visualizes fractal Perlin noise in both 2D and 3D, saving the plots as images.
    Args:
        size (tuple): The dimensions of the noise grid (width, height).
        octaves (int): The number of layers of noise to combine.
        persistence (float): The amplitude reduction factor for each octave.
        amplitude (float): The initial amplitude of the noise.
        lacunarity (float): The frequency multiplier for each octave.
        scale (float): The scale of the noise.
        colors (list[str]): A list of color hex codes or names for the custom colormap.
        bounds (list[float]): A list of boundary values for the colormap normalization.
        show_plots (bool): Whether to display the plots interactively.
        iterNum (int, optional): Iteration number for naming saved files. Defaults to 0.
    Returns:
        None: This function does not return any value. It generates and saves plots.
    Notes:
        - The function normalizes the generated noise values before plotting.
        - Two plots are created: a 2D image and a 3D surface plot.
        - The plots include metadata such as octaves, persistence, amplitude, scale, and lacunarity.
        - The images are saved in the "images/perlin/fractal/" directory with filenames indicating the plot type and iteration number.
    """

    save_filepath = "images/perlin/fractal/fractal_perlin"

    print("Generating fractal Perlin noise...")
    noise = generate_fractal_map(grid_size[0], grid_size[1], scale, octaves, persistence, lacunarity, amplitude, offset_x, offset_y)  # Generate fractal Perlin noise

    # Normalize the noise values to a specified range
    noise = normalize_range(noise)  # Normalize the noise values

    # Define custom colormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # 2D Plot the generated noise
    plt.imshow(noise, cmap=cmap, norm=norm, interpolation='lanczos')
    plt.colorbar()  # Add a color bar to the plot
    plt.title("Fractal Smoothed Perlin Noise")
    plt.text(10, size[0]-100, f"Octaves: {octaves}\nPersistence: {persistence}\nAmplitude: {amplitude}\nScale: {scale}\nLacunarity: {lacunarity}", fontsize=8, color='red', ha='left', va='top')
    plt.savefig(f"{save_filepath}_2d_ {iterNum}.png", dpi=300)  # Save the plot as an image
    print("Plotting 2D fractal Perlin noise...")
    if show_plots:
        plt.show()  # Display the plot
    plt.close()


    # 3D Plot the generated noise
    x = np.linspace(0, size[0]-1, size[0])
    y = np.linspace(0, size[1]-1, size[1])
    x, y = np.meshgrid(x, y)

    # Plotting the 2D Perlin noise in a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    land = ax.plot_surface(x, y, noise, cmap=cmap, norm=norm, edgecolor='none')

    # Add a color bar
    cbar = fig.colorbar(land, ax=ax, boundaries=bounds, ticks=bounds)

    # Labels for the axes
    ax.set_title('Fractal Smoothed Perlin Noise')
    fig.text(0.20, 0.80, f"Octaves: {octaves}\nPersistence: {persistence}\nAmplitude: {amplitude}\nScale: {scale}\nLacunarity: {lacunarity}", fontsize=8, color='red', ha='left', va='top')
    print("Plotting 3D fractal Perlin noise...\n")
    plt.savefig(f"{save_filepath}_3d_{iterNum}.png", dpi=300)  # Save the plot as an image
    if show_plots:
        plt.show()  # Display the plot
    plt.close()



def run_perlin_noise_gaussian(size:tuple, sigmas:list[int], colors:list[str], bounds:list[float], show_plots=True) -> None:
    """
    Generates and visualizes Gaussian-smoothed Perlin noise in both 2D and 3D plots.
    This function creates Perlin noise, applies Gaussian smoothing, normalizes the noise
    values to a given range, and visualizes the noise as a 2D grayscale image and a 3D surface plot.
    The generated plots are saved as image files.

    Args:
        size (tuple): A tuple containing the width and height of the noise grid.
        sigmas (list[int]): A list of sigma values for Gaussian smoothing.
        colors (list[str]): A list of colors for different land types in the visualization.
        bounds (list[float]): A list of boundary values for the land types.
        show_plots (bool, optional): Whether to display the generated plots. Defaults to True.

    Returns:
        None
    """
    save_filepath = "images/perlin/gaussian/gaussian_perlin"
      
    # Define custom colormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # Create a 2D grid of Perlin noise and apply Gaussian smoothing
    print("Generating Perlin noise...")
    noise = generate_perlin_noise(size[0], size[1], scale=0.1)  # Generate Perlin noise

    # Run Gaussian smoothing for each sigma value on singular Perlin noise generation
    for sigma in sigmas:

        # Apply Gaussian smoothing to the noise
        print(f"Applying Gaussian smoothing with sigma={sigma}...")
        noise = gaussian_smooth(noise, sigma)  # Smooth the noise values

        # Normalize the noise values to a specified range
        noise = normalize_range(noise)  # Normalize the noise values

        # 2D Plot the smoothed noise
        plt.imshow(noise, cmap=cmap, norm=norm, interpolation='lanczos')
        plt.colorbar(ticks=bounds, label='Noise Value')  # Add a color bar to the plot
        plt.title(f"Gaussian Smoothed Perlin Noise, sigma={sigma}")
        plt.savefig(f"{save_filepath}_2d_{sigma}.png", dpi=300)  # Save the plot as an image
        print(f"\tPlotting 2D Gaussian, sigma={sigma}, smoothed Perlin noise...")
        if show_plots:
            plt.show()  # Display the plot
        plt.close()

        # 3D Plot the smoothed noise
        x = np.linspace(0, size[0]-1, size[0])
        y = np.linspace(0, size[1]-1, size[1])
        x, y = np.meshgrid(x, y)

        # Plotting the smoothed Perlin noise in a 3D surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        land = ax.plot_surface(x, y, noise, cmap=cmap, norm=norm, edgecolor='none')

        # Add a color bar
        cbar = fig.colorbar(land, ax=ax, boundaries=bounds, ticks=bounds)
        cbar.set_label('Noise Value')

        # Labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Perlin Noise Value')
        ax.set_title(f'Gaussian Smoothed Perlin Noise, sigma={sigma}')
        plt.savefig(f"{save_filepath}_3d_{sigma}.png", dpi=300)  # Save the plot as an image
        print(f"\tPlotting 3D Gaussian, sigma={sigma}, smoothed Perlin noise...")
        if show_plots:
            plt.show()  # Display the plot
        plt.close()

    print(f"Finished generating Gaussian smoothed Perlin noise\n\n")



def ensure_folder_structure():
    """
    Ensures the folder structure for saving images exists.
    If the folders do not exist, they are created.
    """
    folders = [
        "images",
        "images/perlin",
        "images/perlin/fractal",
        "images/perlin/gaussian",
        "images/diamond_square",
        "images/voronoi_biome"
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")



if __name__ == "__main__":
    start_time = time.time()  # Start timing

    # Ensure the folder structure for saving images exists
    ensure_folder_structure()


    # Hyperparameters for Perlin noise grid
    grid_size = (512, 512)


    # Hyperparameters to dictate the land type colors and boundaries
    water_level = 0.30                                                      # Water level for fractal Perlin noise generation
    land_type_boundaries = [0,   0.05,      0.20,   water_level,    0.45,       0.70,   0.90,   1]      # Boundaries for different land types
    land_type_colors = [        'darkblue', "blue", "lightblue", 'darkgreen', 'green','grey', 'white']  # Colors for different land types 


    run_perlin_noise( size = grid_size, \
                    show_plots=False)  # Run the Perlin noise generation and plotting

    params = [0, 0.5, 1, 5, 10, 25, 50, 75, 100]
    run_perlin_noise_gaussian( size=grid_size, \
                            sigmas=params, \
                            colors=land_type_colors, \
                            bounds=land_type_boundaries, \
                            show_plots=False)  # Run the Gaussian-smoothed Perlin noise generation and plotting
    

    # Run fractal Perlin noise with various combinations of parameters
    # octave_values = [1, 3, 5, 7]
    # persistence_values = [0.5, 1.0, 1.5]
    # amplitude_values = [0.25, 0.5, 0.75, 1.0]
    # scale_values = [100, 200, 400, 500]
    # lacunarity_values = [0.5, 1.0]
    # params = len(octave_values) * len(persistence_values) * len(amplitude_values) * len(scale_values) * len(lacunarity_values)
    # current_iteration = 0
    # for scale in scale_values:
    #     for persistence in persistence_values:
    #         for amplitude in amplitude_values:
    #             for octaves in octave_values:
    #                 for lacunarity in lacunarity_values:
    #                     current_iteration += 1
    #                     print(f"Iteration {current_iteration} of {params} - Progress: {(current_iteration / params) * 100:.2f}% complete")
    #                     print(f"Running fractal Perlin noise with octaves={octaves}, persistence={persistence}, amplitude={amplitude}, scale={scale}, lacunarity={lacunarity}")
    #                     run_perlin_noise_fractal(
    #                         size=grid_size,
    #                         octaves=octaves,
    #                         persistence=persistence,
    #                         amplitude=amplitude,
    #                         scale=scale,
    #                         lacunarity=lacunarity,
    #                         colors=land_type_colors,
    #                         bounds=land_type_boundaries,
    #                         show_plots=False,
    #                         offset_x=random.random() * 100,
    #                         offset_y=random.random() * 100,
    #                         iterNum=current_iteration
    #                     )

    # params = [
    #     #octaves, persistence, amplitude, lacunarity, scale
    #     (1,         1.0,        0.25,        0.5,      500),
    #     (1,         1.0,        0.50,        0.5,      500),
    #     (1,         1.0,        0.75,        0.5,      500),
    #     (1,         1.0,        1.00,        0.5,      500),
    # ]
    # for i, (octaves, persistence, amplitude, lacunarity, scale) in enumerate(params):
    #     print(f"Iteration {i} of {len(params)} - Progress: {(i / len(params)) * 100:.0f}% complete")
    #     print(f"Running fractal Perlin noise with octaves={octaves}, persistence={persistence}, amplitude={amplitude}, scale={scale}, lacunarity={lacunarity}")
    #     run_perlin_noise_fractal(
    #         size=grid_size,
    #         octaves=octaves,
    #         persistence=persistence,
    #         amplitude=amplitude,
    #         scale=scale,
    #         lacunarity=lacunarity,
    #         colors=land_type_colors,
    #         bounds=land_type_boundaries,
    #         show_plots=False,
    #         offset_x=random.random() * 100,
    #         offset_y=random.random() * 100,
    #         iterNum=i
    # )
        

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total time taken: {int(minutes)} minutes and {seconds:.2f} seconds")
    print(f"Average time per iteration: {total_time / len(params):.2f} seconds\n\n")
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
        

    # # Find the grid cell coordinates
    # x_i:int = int(np.floor(x)) & 255  # Wrap to [0, 255]
    # y_i:int = int(np.floor(y)) & 255  # Wrap to [0, 255]


    # # Find the relative coordinates within the cell
    # x_f:float = x - np.floor(x)  # Fractional part of x
    # y_f:float = y - np.floor(y)  # Fractional part of y


    # # Fade the curves for smooth interpolation
    # u = fade(x_f)
    # v = fade(y_f)


    # # Hash coordinates of the cube corners
    # A = permutations[x_i] + y_i
    # B = permutations[x_i + 1] + y_i
    # # ba = permutations[x_i] + y_i + 1
    # # bb = permutations[x_i + 1] + y_i + 1
        

    # # Compute the gradient at each corner of the cube
    # grad_1 = grad(permutations[A], x_f, y_f)
    # grad_2 = grad(permutations[B], x_f - 1, y_f)
    # # grad_ba = grad(permutations[ba], x_f, y_f - 1)
    # # grad_bb = grad(permutations[bb], x_f - 1, y_f - 1)
    # grad_3 = grad(permutations[A+1], x_f, y_f - 1)
    # grad_4 = grad(permutations[B+1], x_f - 1, y_f - 1)


    # # Interpolate between the gradients
    # x1 = interpolate(grad_1, grad_2, u)  # Interpolate along x-axis
    # x2 = interpolate(grad_3, grad_4, u)  # Interpolate along x-axis
    # result = interpolate(x1, x2, v)  # Interpolate along y-axis

    # return result
    # ==================================================================================
    X = int(np.floor(x)) & 255
    Y = int(np.floor(y)) & 255
        
    # Find relative x, y of point in square
    x -= np.floor(x)
    y -= np.floor(y)
        
    # Compute fade curves for x and y
    u = fade(x)
    v = fade(y)
        
    # Hash coordinates of the 4 square corners
    A = permutations[X] + Y
    B = permutations[X + 1] + Y
    C = permutations[X] + Y + 1
    D = permutations[X + 1] + Y + 1
        
    # Calculate noise contributions from each corner
    g1 = grad(permutations[A], x, y)
    g2 = grad(permutations[B], x-1, y)
    g3 = grad(permutations[C+1], x, y-1)
    g4 = grad(permutations[D+1], x-1, y-1)
        
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


    
def generate_fractal_noise(width: int, height: int, octaves: int, persistence: float, amplitude: float, scale: float) -> np.ndarray[float]:
    """
    Generate a 2D grid of fractal Perlin noise by combining multiple octaves of Perlin noise.

    Args:
        width (int): The width of the noise grid.
        height (int): The height of the noise grid.
        octaves (int): The number of octaves for the fractal noise.
        persistence (float): The persistence value controlling the amplitude of each octave.
        amplitude (float): The initial amplitude of the noise.
        scale (float): The scale factor for the noise.

    Returns:
        np.ndarray[float]: A 2D array of fractal Perlin noise values.
    
    Note:
        This function does not normalize the output. Use `normalize_range` if normalization is required.
    """
    perms = generate_permutation()  # Generate the permutation array
    noise_grid = np.zeros((height, width), dtype=float)

    for octave in range(octaves):
        print(f"\tGenerating octave {octave + 1} of {octaves}...")
        frequency = scale * (2 ** octave)
        octave_amplitude = amplitude * (persistence ** octave)

        for y in range(height):
            for x in range(width):
                noise_grid[y, x] += noise(x * frequency, y * frequency, perms) * octave_amplitude

    return noise_grid

# =======================================================================================================================================================

def octave_noise(x, y, permutations, octaves=1, persistence=0.5, lacunarity=2.0):
    """Generate Perlin noise with multiple octaves for more natural patterns.
    
    Args:
        x, y: Coordinates to generate noise at
        octaves: Number of octaves to sum
        persistence: How much each octave contributes to the final result
        lacunarity: How much the frequency increases with each octave
        
    Returns:
        A float value that is the sum of all octaves, roughly in range [-1, 1]
    """
    total = 0
    frequency = 1
    amplitude = 1
    max_value = 0  # Used for normalizing result
    
    for _ in range(octaves):
        total += noise(x * frequency, y * frequency, permutations) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
        
    # Normalize the result
    return total / max_value if max_value > 0 else 0



def generate_noise_map(width, height, scale=0.1, octaves=4, persistence=0.5, lacunarity=2.0):
    """Generate a 2D noise map.
    
    Args:
        width, height: Dimensions of the noise map
        scale: Level of zoom/detail in the noise
        octaves: Number of layers of noise
        persistence: How much each octave contributes
        lacunarity: How much the frequency increases with each octave
        
    Returns:
        2D list containing noise values typically in range [-1, 1]
    """
    permutations = generate_permutation()  # Generate the permutation array
    noise_map = np.zeros((height, width), dtype=float)
    
    # Offsets can be used to sample different areas of the noise
    offset_x = random.random() * 100
    offset_y = random.random() * 100
    
    for y in range(height):
        for x in range(width):
            # Scale coordinates and generate noise
            nx = (x + offset_x) * scale
            ny = (y + offset_y) * scale
            noise_map[y][x] = octave_noise(nx, ny, permutations, octaves, persistence, lacunarity)
            
    return noise_map






# =======================================================================================================================================================

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



def run_perlin_noise_fractal(size:tuple, octaves:int, persistence:float, amplitude:float, scale:float, colors:list[str], bounds:list[float], show_plots:bool, iterNum:int=0) -> None:
    """
    Generates and visualizes fractal Perlin noise in both 2D and 3D plots.
    This function creates fractal Perlin noise using the specified parameters, normalizes the noise
    values to a given range, and visualizes the noise as a 2D grayscale image and a 3D surface plot.
    The generated plots are saved as image files.\
    
    Args:
        size (tuple): A tuple containing the width and height of the noise grid.
        octaves (int): The number of octaves for the fractal noise generation.
        persistence (float): The persistence value controlling the amplitude of each octave.
        amplitude (float): The initial amplitude of the noise.
        scale (float): The scale factor for the noise.
        colors (list[str]): A list of colors for different land types in the visualization.
        bounds (list[float]): A list of boundary values for the land types.
        show_plots (bool): Whether to display the generated plots.
        iterNum (int, optional): The iteration number for saving unique filenames. Defaults to 0.

    Returns:
        None
    """
    save_filepath = "images/perlin/fractal/fractal_perlin"

    print("Generating fractal Perlin noise...")
    # noise = generate_fractal_noise(grid_size[0], grid_size[1], octaves, persistence, amplitude, scale)  # Generate fractal Perlin noise
    noise = generate_noise_map(grid_size[0], grid_size[1], scale, octaves, persistence)  # Generate fractal Perlin noise

    # Normalize the noise values to a specified range
    # noise = normalize_range(noise)  # Normalize the noise values

    # Define custom colormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # 2D Plot the generated noise
    plt.imshow(noise, cmap=cmap, norm=norm, interpolation='lanczos')
    plt.colorbar()  # Add a color bar to the plot
    plt.title("Fractal Smoothed Perlin Noise")
    plt.text(10, size[0]-75, f"Octaves: {octaves}\nPersistence: {persistence}\nAmplitude: {amplitude}\nScale: {scale}", fontsize=8, color='red', ha='left', va='top')
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
    cbar.set_label('Noise Value')

    # Labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Perlin Noise Value')
    ax.set_title('Fractal Smoothed Perlin Noise')
    fig.text(0.20, 0.80, f"Octaves: {octaves}\nPersistence: {persistence}\nAmplitude: {amplitude}\nScale: {scale}", fontsize=8, color='red', ha='left', va='top')
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
    grid_size = (256, 256)


    # Hyperparameters to dictate the land type colors and boundaries
    water_level = 0.30                                     # Water level for fractal Perlin noise generation
    land_type_boundaries = [0, water_level, 0.45, 0.70, 0.90, 1]  # Boundaries for different land types
    land_type_colors = ['blue', 'green', 'darkgreen','grey', 'white']  # Colors for different land types 


    # run_perlin_noise( size = grid_size, \
    #                 show_plots=False)  # Run the Perlin noise generation and plotting

    # run_perlin_noise_gaussian( size=grid_size, \
    #                         sigmas=[0, 0.5, 1, 10, 25, 50, 75, 100, 250], \
    #                         colors=land_type_colors, \
    #                         bounds=land_type_boundaries, \
    #                         show_plots=False)  # Run the Gaussian-smoothed Perlin noise generation and plotting




    run_perlin_noise_fractal( size=grid_size, \
                            octaves=7, \
                            persistence=0.5, \
                            amplitude=1.75, \
                            scale=400, \
                            colors=land_type_colors, \
                            bounds=land_type_boundaries, \
                            show_plots=False)  # Run the fractal Perlin noise generation and plotting



# Run fractal Perlin noise with various combinations of parameters
# octave_values = [2, 4, 6, 8, 10]
# persistence_values = [0.3, 0.5, 0.7]
# amplitude_values = [0.4, 0.6, 0.8]
# scale_values = [0.1, 0.01, 0.001, 0.0001]

# total_iterations = len(octave_values) * len(persistence_values) * len(amplitude_values) * len(scale_values)
# current_iteration = 0

# for scale in scale_values:
#     for persistence in persistence_values:
#         for amplitude in amplitude_values:
#             for octaves in octave_values:
#                 current_iteration += 1
#                 print(f"Iteration {current_iteration} of {total_iterations} - Progress: {(current_iteration / total_iterations) * 100:.2f}% complete")
#                 print(f"Running fractal Perlin noise with octaves={octaves}, persistence={persistence}, amplitude={amplitude}, scale={scale}")
#                 run_perlin_noise_fractal(
#                     width=width,
#                     height=height,
#                     octaves=octaves,
#                     persistence=persistence,
#                     amplitude=amplitude,
#                     scale=scale,
#                     colors=land_type_colors,
#                     bounds=land_type_boundaries,
#                     show_plots=False,
#                     iterNum=current_iteration
#                 )

# end_time = time.time()  # End timing
# total_time = end_time - start_time
# minutes, seconds = divmod(total_time, 60)
# print(f"Total time taken: {int(minutes)} minutes and {seconds:.2f} seconds")
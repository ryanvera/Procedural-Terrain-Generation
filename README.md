# Procedural Terrain Generation<br/>
## Perlin Noise, Diamond Square, and Voronoi Biome
Ryan Vera<br/>
Qitong Liu

CS 6366 - Computer Graphics


## Introduction
This project implements and compares three algorithms commonly used in procedural generation:
- **Perlin Noise**
- **Diamond Square**
- **Voronoi Biome**

The goal is to analyze their performance, visual output, and use cases in computer graphics.

## File Structure
```
/Project
│
├── perlin_noise.py       # Implementation of the Perlin Noise algorithm
├── diamond_square.py     # Implementation of the Diamond Square algorithm
├── voronoi_biome.py      # Implementation of the Voronoi Biome algorithm
├── README.md             # Project documentation
├── requirements.txt      # Required packages
├── images/               # Folder for generated outputs
│   ├── perlin/           # Perlin Noise output images
│   │   ├── fractal/      # Fractal-based Perlin Noise images
│   │   └── gaussian/     # Gaussian-smoothed Perlin Noise images
│   ├── diamond_square/   # Diamond Square output images
│   └── voronoi/          # Voronoi Biome output images
```


### Perlin Noise
Perlin Noise is a gradient noise function developed by Ken Perlin, widely used in procedural texture generation. It produces smooth, continuous patterns that resemble natural phenomena such as clouds, terrain, or wood grain. The algorithm works by interpolating random gradients at grid points and blending them to create a coherent noise pattern. It is computationally efficient and supports multi-octave layering to achieve more complex textures. 

In this implementation, smoothing was performed using a fractal method, which combines multiple layers of Perlin Noise at different frequencies and amplitudes. This approach enhances the visual complexity and realism of the generated patterns.

### Diamond Square
The Diamond Square algorithm is a fractal-based technique for generating heightmaps, often used in terrain generation. It starts with a grid of points, where corner values are initialized. The algorithm alternates between two steps:
1. **Diamond Step**: Calculates the midpoint of each square by averaging its corners and adding a random offset.
2. **Square Step**: Calculates the midpoint of each edge by averaging the two adjacent corners and adding a random offset.
This process is repeated recursively, with the random offset decreasing at each iteration to create realistic terrain with varying levels of detail.

### Voronoi Biome
Voronoi Biome generation is based on Voronoi diagrams, which partition space into regions based on the proximity to a set of seed points. Each region corresponds to the area closest to a specific seed. This method is commonly used in procedural generation to simulate natural phenomena such as cellular structures, biome distribution, or settlement layouts. By assigning different properties to each region, Voronoi diagrams can create diverse and visually appealing patterns for use in games or simulations.

## References
1.  
2. 
3. 

## How to Run
1. Clone the repository.
2. Run each algorithm file individually to generate outputs.
3. Compare the results visually

### Output File Naming
The `run_perlin_noise_gaussian` function generates both 2D and 3D plots of Gaussian-smoothed Perlin noise. The output file names follow the structure:

- **2D Plot**: `gaussian_perlin_2d_<sigma>.png`
- **3D Plot**: `gaussian_perlin_3d_<sigma>.png`

Here, `<sigma>` represents the standard deviation used for Gaussian smoothing. These files are saved in the `images/` directory.

## License
[MIT License](LICENSE)
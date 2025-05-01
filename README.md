# Procedural Terrain Generation - Perlin Noise, Diamond Square, and Voronoi Biome
CS 6366 - Computer Graphics  <br/>
Qitong Liu  
Ryan Vera


## Introduction
Procedural terrain generation is a fundamental technique in computer graphics, enabling the creation of realistic and diverse landscapes for applications such as video games, simulations, and virtual environments. By leveraging mathematical algorithms, procedural generation eliminates the need for manual modeling, allowing for the efficient creation of large-scale, detailed terrains. This project focuses on implementing and comparing three widely used algorithms in procedural terrain generation: **Perlin Noise**, **Diamond Square**, and **Voronoi Biome**. Each algorithm offers unique approaches to terrain generation, ranging from smooth gradient-based patterns to fractal heightmaps and region-based biome partitioning.

The primary objective of this project is to explore the strengths and limitations of these algorithms by analyzing their performance, visual output, and potential use cases. Through this comparison, we aim to provide insights into how these techniques can be applied to generate realistic and visually appealing terrains. Additionally, the project serves as an educational resource for understanding the principles and parameters that govern procedural terrain generation. By experimenting with these algorithms, users can gain a deeper appreciation for the mathematical and computational foundations of terrain modeling in computer graphics.

## Algorithm Descriptions
The following section provides a detailed explanation of the three algorithms implemented in this project. Each algorithm is described in terms of its purpose, methodology, and key parameters. By understanding these algorithms, readers can gain insights into their applications in procedural terrain generation and their respective strengths and limitations.

---

### Perlin Noise
Perlin Noise is a gradient noise function developed by Ken Perlin, widely used in procedural texture generation. It produces smooth, continuous patterns that resemble natural phenomena such as clouds, terrain, or wood grain. The algorithm works by interpolating random gradients at grid points and blending them to create a coherent noise pattern. It is computationally efficient and supports multi-octave layering to achieve more complex textures. 

In this implementation, smoothing was performed using a fractal method, which combines multiple layers of Perlin Noise at different frequencies and amplitudes. This approach enhances the visual complexity and realism of the generated patterns.

#### Fractal Smoothing Parameters
- **Octaves**: The number of layers of Perlin Noise combined to create the final texture. Higher values result in more detail.
- **Persistence**: Controls the amplitude of each successive octave. A lower value reduces the contribution of higher-frequency layers.
- **Lacunarity**: Determines the frequency of each successive octave. Higher values increase the frequency, creating finer details.
- **Amplitude**: Controls the intensity of the noise pattern. Higher values increase the contrast between peaks and valleys, creating more pronounced features, while lower values produce smoother, less distinct patterns. Adjusting the amplitude allows for fine-tuning the visual impact of the generated texture.
- **Scale**: Determines the spacing between the gradients in the Perlin Noise grid. Smaller values result in more compressed patterns with finer details, while larger values produce broader, smoother patterns. Adjusting the scale allows for control over the level of detail in the generated texture.

#### Gaussian Smoothing Parameters
- **Sigma**: Represents the standard deviation of the Gaussian function used for smoothing Perlin noise. 
    A higher sigma value results in smoother noise with less detail, while a lower sigma value retains more fine-grained details.

---

### Diamond Square
The Diamond Square algorithm is a fractal-based technique for generating heightmaps, often used in terrain generation. It starts with a grid of points, where corner values are initialized. The algorithm alternates between two steps:
1. **Diamond Step**: Calculates the midpoint of each square by averaging its corners and adding a random offset.
2. **Square Step**: Calculates the midpoint of each edge by averaging the two adjacent corners and adding a random offset.
This process is repeated recursively, with the random offset decreasing at each iteration to create realistic terrain with varying levels of detail.

#### Parameters
- **Grid Size**: Determines the resolution of the heightmap.
- **Randomness**: Controls the variation in terrain features.

---

### Voronoi Biome
Voronoi Biome generation is based on Voronoi diagrams, which partition space into regions based on the proximity to a set of seed points. Each region corresponds to the area closest to a specific seed. This method is commonly used in procedural generation to simulate natural phenomena such as cellular structures, biome distribution, or settlement layouts. By assigning different properties to each region, Voronoi diagrams can create diverse and visually appealing patterns for use in games or simulations.

#### Parameters
- **Seed Points**: Number and distribution of regions.
- **Region Properties**: Attributes assigned to each region.

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
|   ├── true_perlin       # Contains example images of Perlin Noise generated by a noise library 
│   ├── diamond_square/   # Diamond Square output images
│   └── voronoi/          # Voronoi Biome output images
```

## How to Run
1. Clone the repository to your local machine.
2. Install the required Python packages using:
   ```
   pip install -r requirements.txt
   ```
3. Run the individual algorithm files to generate terrain outputs:
   - `perlin_noise.py`
   - `diamond_square.py`
   - `voronoi_biome.py`
4. View the generated images in the `images/` directory.

## Output File Naming
Generated files follow a structured naming convention:
- **Perlin Noise (Gaussian)**:
  - 2D Plot: `gaussian_perlin_2d_<sigma>.png`
  - 3D Plot: `gaussian_perlin_3d_<sigma>.png`
- **Perlin Noise (Fractal)**:
  - 2D Plot: `fractal_perlin_2d_<iteration number>.png`
  - 3D Plot: `fractal_perlin_3d_<iteration number>.png`
- **Diamond Square**: `diamond_square_<grid_size>.png`
- **Voronoi Biome**: `voronoi_biome_<seed_count>.png`

## License
[MIT License](LICENSE)

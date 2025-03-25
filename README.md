# Parallel Algorithms Implementation

This repository contains implementations of parallel algorithms developed as part of the Parallel Computing course. It includes parallel implementations of merge sort using OpenMP and tiled matrix convolution using CUDA.

| Student |
|-------- |
| [Capodanno Mario](https://github.com/MarioCapodanno) |
| [Grillo Valerio](https://github.com/Valegrl) |
| [Lovino Emanuele](https://github.com/EmanueleLovino) |

## Repository Structure

The repository is organized into the following directories:

### `parallel_merge_sort`
This folder contains a parallel implementation of the merge sort algorithm using OpenMP.

- **`mergesort.cpp`**: C++ source code of the parallel merge sort implementation.
- **`report_challenge_1.pdf`**: A report detailing the implementation and performance analysis of the parallel merge sort.

### `tiled_matrix_convolution`
This folder contains a CUDA implementation of both tiled and non-tiled 2D convolution.

- **`cuda_convolution_2D.ipynb`**: Jupyter Notebook containing the CUDA implementation of tiled and non-tiled 2D convolution.
- **`report_challenge_2.pdf`**: A report detailing the implementation and performance analysis of the tiled and non-tiled 2D convolution.

## Implemented Tasks

### Parallel Merge Sort
The objective of this challenge was to parallelize the merge sort algorithm using OpenMP. The parallel implementation was tested on arrays ranging from **1 million to 100 million elements**, and performance results were analyzed.

### Tiled Matrix Convolution
This challenge implements both tiled and non-tiled 2D convolution using CUDA. The CUDA code is optimized to run on an **NVIDIA Tesla T4 GPU**.

## How to Use
### Running Parallel Merge Sort
1. Compile the C++ code with OpenMP support:
   ```sh
   g++ -fopenmp mergesort.cpp -o mergesort
   ```
2. Run the executable:
   ```sh
   ./mergesort
   ```

### Running CUDA Convolution
1. Open the Jupyter Notebook `cuda_convolution_2D.ipynb`.
2. Execute the notebook on a machine with an NVIDIA GPU and CUDA support.

## License
This repository is for educational purposes as part of the Parallel Computing course.


# Edge Detection Filter

## Introduction

In this project, we implemented an [Edge Detection Filter based on Sobel Filter](https://en.wikipedia.org/wiki/Sobel_operator). The project’s core is based on SIMD -or, to be more precise, SIMT- computations using NVIDIA CUDA. The project features both CLI and GUI. The GUI is written in Python, uses the Tkinter library, and communicates with the CUDA core, which is written in C++ using a shell.

For more information regarding the implementation and analysis of the result, read `Document.pdf`.

## Compilation

To be able to compile the core, you should have previously installed [OpenCV](https://opencv.org/) and [NVIDIA Cuda compiler](https://developer.nvidia.com/cuda-downloads).

To compile the core, run the following command:

```bash
nvcc main.cu `pkg-config opencv --cflags --libs` -o main.out
```

#### Note
Based on the version of OpenCV and the system you are using, you may need to change the command like this:

```bash
nvcc main.cu -arch=sm_86 `pkg-config opencv4 --cflags --libs` -o main.out
```

Also, note that `sm_86` is used for a GPU with 8.6 compute capability. You should change it based on your [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus). You can find your GPU Compute Capability using the following command:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1
```


## Run

### GUI

Put the `main.out` file in the same directory as `main.py` and run the following command:


```bash
python GUI/main.py
```

### CLI


Run `main.out` with the following arguments:

```bash
./main.out [image_path] [alpha] [beta] [thresh]
```

- *alpha*: Desired Contrast Value
- *beta*: Desired Brigtness Value
- *thresh*: Desired Sobel Filter Threshold

## Demo

![Demo](demo.gif)

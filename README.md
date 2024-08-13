# CUDAatScaleForTheEnterpriseCourseProjectTemplate
This is a template for the course project for the CUDA at Scale for the Enterprise

## Project Description

This project was created as a playground for investigating various filtering and image processing capabilities of the CUDA NPP (NVIDIA 2D Image and Signal Processing Performance Primitives) Library. Currently, the project provides an example usecasefor the following NPP functions:

- AddC from Image Arithmetic And Logical Operations
- Erode from Image Morphological Operations

The project allows to choose the input image file in BMP or PGM format, specify the needed filter and provide the filename or directory for the output file. Currently, the project allows processing only one image, since there is a problem with NPP kernel execution, which fails when you try to run the same kernel again. The project requires a Coursera Lab environment to execute since it provides the configured CUDA environment and doesn't require additional configuration, which currently is out of the scope of this project.

The project structure follows the template from https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate.

## How to run it

Run the `./bin/x86_64/linux/release/imageProcessing.exe` or the `./run.sh` file. Available flags:

- `--addNumber`: What should be added to the image with the AddC function. The default value is 100.
- `--input`: The path for the input file. The default is the ./data/Lena.pgm image.

Makefile is written to provide an easy way to delete an already existing compiled executables (`make clean`), compile the source code into executable file (`make build`) and run it (`make run`). The latter one also compiles it before running it.

## Results

![Original image](https://github.com/Lehohel/CUDAatScaleForTheEnterprise/blob/main/images/Lena.jpg)

Fig 1. The original test image.

![Image after add function](https://github.com/Lehohel/CUDAatScaleForTheEnterprise/blob/main/images/Lena_add.jpg)

Fig 2. The image after the AddC function was applied to it with the default value.

![Eroded imgage](https://github.com/Lehohel/CUDAatScaleForTheEnterprise/blob/main/images/Lena_erode.jpg)

Fig 3. The image after the Erode function is applied to it.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```images/```
Images in browser viewable format. It contain the output of the code in jpg format.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.

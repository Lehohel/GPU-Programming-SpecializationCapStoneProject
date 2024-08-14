# GPU Specialization Capstone Project: Nonlinear optical Kerr effect
This repository contains the code base for the solution of the "Peer-graded Assignment: GPU Specialization Capstone Project" on Coursera.

## Project Description

In this project, the [4th order Runge-Kutta differential equation solver](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) was implemented to numerically solve the [nonlinear Schrödinger equation](https://en.wikipedia.org/wiki/Nonlinear_Schr%C3%B6dinger_equation) in dispersionless case. This problem has a simple analytical solution, which was compared to the result of the numrical simulation.

The equation describes an optical pulse's evolution during the propagation in nonlinear medium (i.e. optical fiber). The code calculates this evolution on a given temporal grid.

The code utalizes pinned memory, shared memory and constant memory for enchanced performance and does the calculations on cuFloatComplex arrays.

The project structure follows the template from https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate.

## How to run it

Run the `./bin/KerrEffect.exe` or the `./run.sh` file. Available flags:

- `--threadsPerBlock`:  The number of thread in a block on the device. The default value is 256
- `--notp`:  The number of points in the temporal domain. The default value is 2048
- `--thickness`:  The thickness of the material in millimeter. The default value is 1.0
- `--intensity`:  The intensity of the optical pulse in TW/cm2. The default value is 10.0
- `--tau`:  The pulse duaration in femtosecond. The default value is 30.0
- `--wl0`:  The cetral wavelength of the optical pulse in nanometer. The default value is 800.0
- `--dt`:  The temporal distance between two datapoints in femtosecond. The default value is 0.1
- `--output`:  The name of the output csv file. The file is created in the `./data` folder, and the `.csv` extension is appended to it. The default value is `output`

Makefile is written to provide an easy way to delete an already existing compiled executables (`make clean`), compile the source code into executable file (`make build`) and run it (`make run`).

The `./run.sh` bash script provides examples for running the code with default arguments and with non-default arguments.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder contains the result of the numerical simulation in `.csv` format.

```Common/```
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

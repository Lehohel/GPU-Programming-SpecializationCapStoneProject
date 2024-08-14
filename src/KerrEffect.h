#ifndef KER_REFFECT_H
#define KER_REFFECT_H

#include <string.h>
#include <iostream>
#include <complex>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuComplex.h>

// typedef float2 Complex;
// using namespace std::complex_literals;

__global__ void oneStepRK4(cuFloatComplex *d_field, float *d_b, float *d_c, int numElements, float stepSize);
__device__ void diffEqEval(cuFloatComplex field, cuFloatComplex *res);
__device__ void RK4Helper(cuFloatComplex fields[6], float stepSize);
__host__ bool printfCUDAinfo(int argc, char *argv[]);
__host__ void readArgument(std::string name, int *var, int argc, char *argv[]);
__host__ void readArgument(std::string name, float *var, int argc, char *argv[]);
__host__ void readArgument(std::string name, std::string *var, int argc, char *argv[]);
__host__ cuFloatComplex electricField(float t, float pulseDuration, float intensity);
__host__ cuFloatComplex *allocateDeviceMemory(int numberOfTemporalPoints);
__host__ void copyFromHostToDevice(cuFloatComplex *h_a, cuFloatComplex *d_a, int numberOfTemporalPoints);
__host__ void executeKernel(cuFloatComplex *d_field0, cuFloatComplex *d_fieldFinal, int threadsPerBlock, int notp,
                            int nopp, float thickness, float centralWavelength, float timeStep);
__host__ void copyFromDeviceToHost(cuFloatComplex *d_a, cuFloatComplex *h_a, int numberOfTemporalPoints);
__host__ void deallocateMemory(cuFloatComplex *d_a);
__host__ void cleanUpDevice();
__host__ cuFloatComplex* analyticalSolution(cuFloatComplex *field, int notp, float nonLinearRefIndex,
                                           float centralWavelength, float linearRefIndex, float thickness);
__host__ void saveData(std::string fileName, float *time, cuComplex *field0, cuComplex *fieldFinal, cuComplex *analytical, int notp);
__host__ float calculateDifference(cuFloatComplex *numeric, cuFloatComplex *analytic, int notp);
__host__ int main(int argc, char *argv[]);

const float speedOfLight = 299.792458f; // Speed of light in vacuum in nm/fs
const float pi = 3.14159265359f;

__constant__ cuFloatComplex d_diffEqFactor;

#endif
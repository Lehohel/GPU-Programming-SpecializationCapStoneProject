#include "KerrEffect.h"

__global__ void oneStepRK4(cuFloatComplex *d_field, cuFloatComplex *d_res, int numberOfTemporalPoints, float stepSize)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numberOfTemporalPoints)
  {
    // int tid = threadIdx.x;
    __shared__ cuFloatComplex dataPoints[6];
    dataPoints[0] = d_field[i];
    diffEqEval(dataPoints[0], &dataPoints[1]);
    diffEqEval(cuCaddf(dataPoints[0], cuCmulf(dataPoints[1], make_cuFloatComplex(stepSize / 2, 0.0))), &dataPoints[2]);
    diffEqEval(cuCaddf(dataPoints[0], cuCmulf(dataPoints[2], make_cuFloatComplex(stepSize / 2, 0.0))), &dataPoints[3]);
    diffEqEval(cuCaddf(dataPoints[0], cuCmulf(dataPoints[3], make_cuFloatComplex(stepSize, 0.0))), &dataPoints[4]);
    RK4Helper(dataPoints, stepSize);
    d_res[i] = dataPoints[5];
  }
}

__device__ void diffEqEval(cuFloatComplex field, cuFloatComplex *res){
  float intens = field.x * field.x + field.y * field.y;
  *res = cuCmulf(make_cuFloatComplex(intens, 0.0), cuCmulf(field, d_diffEqFactor));
}

__device__ void RK4Helper(cuFloatComplex dataPoints[6], float stepSize)
{
  dataPoints[5].x = dataPoints[0].x + (stepSize / 6) * (dataPoints[1].x + 2 * dataPoints[2].x + 2 * dataPoints[3].x + dataPoints[4].x);
  dataPoints[5].y = dataPoints[0].y + (stepSize / 6) * (dataPoints[1].y + 2 * dataPoints[2].y + 2 * dataPoints[3].y + dataPoints[4].y);
}

__host__ bool printfCUDAinfo(int argc, char *argv[])
{
  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  printf("\n");
  return bVal;
}

__host__ void readArgument(std::string name, int *var, int argc, char *argv[])
{
  char *nameChar;
  if (checkCmdLineFlag(argc, (const char **)argv, name.c_str()))
  {
    getCmdLineArgumentString(argc, (const char **)argv, name.c_str(), &nameChar);
    *var = std::stoi(nameChar);
    printf("   CL argument --%s was read, and the value was set to: %d\n", name.c_str(), *var);
  }
  else{
    printf("No CL argument --%s was read, and the value was set to: %d (default)\n", name.c_str(), *var);
  }
}

__host__ void readArgument(std::string name, float *var, int argc, char *argv[])
{
  char *nameChar;
  if (checkCmdLineFlag(argc, (const char **)argv, name.c_str()))
  {
    getCmdLineArgumentString(argc, (const char **)argv, name.c_str(), &nameChar);
    *var = std::stof(nameChar);
    printf("   CL argument --%s was read, and the value was set to: %f\n", name.c_str(), *var);
  }
  else
  {
    printf("No CL argument --%s was read, and the value was set to: %f (default)\n", name.c_str(), *var);
  }
}

__host__ void readArgument(std::string name, std::string *var, int argc, char *argv[])
{
  char *nameChar;
  if (checkCmdLineFlag(argc, (const char **)argv, name.c_str()))
  {
    getCmdLineArgumentString(argc, (const char **)argv, name.c_str(), &nameChar);
    *var = (std::string)(nameChar);
    printf("   CL argument --%s was read, and the value was set to: %s\n", name.c_str(), (*var).c_str());
  }
  else
  {
    printf("No CL argument --%s was read, and the value was set to: %s (default)\n", name.c_str(), (*var).c_str());
  }
}

__host__ cuFloatComplex electricField(float t, float pulseDuration, float intensity)
{
  cuFloatComplex f;
  f.x = sqrt(intensity) * exp(-1 * pow(t / pulseDuration, 2) * 2 * log(2));
  f.y = 0;
  return f;
}

__host__ cuFloatComplex * allocateDeviceMemory(int numberOfTemporalPoints)
{
  // Allocate the device input vector A
  cuFloatComplex *d_a = NULL;
  size_t size = numberOfTemporalPoints * sizeof(cuFloatComplex);

  cudaError_t err = cudaMalloc(&d_a, size);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return d_a;
}

__host__ void copyFromHostToDevice(cuFloatComplex *h_a, cuFloatComplex *d_a, int numberOfTemporalPoints)
{
  size_t size = numberOfTemporalPoints * sizeof(cuFloatComplex);

  cudaError_t err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void executeKernel(cuFloatComplex *d_field0, cuFloatComplex *d_fieldFinal, int threadsPerBlock,
                            int notp, int nopp, float thickness, float centralWavelength, float timeStep)
{
  cuFloatComplex *A = d_field0;
  cuFloatComplex *B = d_fieldFinal;
  // Launch the search CUDA Kernel
  int blocksPerGrid = (notp + threadsPerBlock - 1) / threadsPerBlock;
  float stepSize = thickness / nopp;
  for (int i = 0; i < nopp; i++){
    oneStepRK4<<<blocksPerGrid, threadsPerBlock>>>(A, B, notp, stepSize);
    // Switch the input / output arrays instead of copying them around
    A = A == d_field0 ? d_fieldFinal : d_field0;
    B = B == d_field0 ? d_fieldFinal : d_field0;
  }

  d_fieldFinal = A;

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void copyFromDeviceToHost(cuFloatComplex *d_a, cuFloatComplex *h_a, int numberOfTemporalPoints)
{
  size_t size = numberOfTemporalPoints * sizeof(cuFloatComplex);

  cudaError_t err = cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void deallocateMemory(cuFloatComplex *d_a)
{

  cudaError_t err = cudaFree(d_a);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaError_t err = cudaDeviceReset();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ cuFloatComplex* analyticalSolution(cuFloatComplex *field, int notp, float nonLinearRefIndex,
                                           float centralWavelength, float linearRefIndex, float thickness)
{
  cuFloatComplex *result = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * notp);
  // cuFloatComplex result[notp];
  float diffEqFactor = nonLinearRefIndex * (2 * pi / centralWavelength * 1e6) / linearRefIndex;
  for (int i = 0; i < notp; i++)
  {
    float intensity = field[i].y * field[i].y + field[i].x * field[i].x;
    float phi0 = atan2(field[i].y, field[i].x);
    float phi = phi0 - diffEqFactor * thickness * intensity;
    result[i].x = sqrt(intensity) * cos(phi);
    result[i].y = sqrt(intensity) * sin(phi);
  }
  return result;
}

__host__ void saveData(std::string fileName, float *time, cuComplex *field0, cuComplex *fieldFinal, cuComplex *analytical, int notp){
  std::ofstream myfile;
  myfile.open("./data/"+fileName+".csv");
  myfile << "Time [fs],Original field real,Original field imaginary,RK4 field real,RK4 field imaginary,Analytical real,Analytical imaginary\n";
  for (int i = 0; i < notp; i += 1)
  {
    myfile << time[i] << ',' << field0[i].x << ',' << field0[i].y << ',' << fieldFinal[i].x << ',' << fieldFinal[i].y << ',' << analytical[i].x << ',' << analytical[i].y << '\n';
  }
  myfile.close();
}

__host__ float calculateDifference(cuFloatComplex *numeric, cuFloatComplex *analytic, int notp)
{
  printf("Comparing analytical and numerical solutions\n");
  int totalsquaredifference = 0.0f;

  for (int i = 0; i < notp; i++)
  {
    cuFloatComplex difference;
    difference.x = analytic[i].x - numeric[i].x;
    difference.y = analytic[i].y - numeric[i].y;
    totalsquaredifference += difference.x * difference.x + difference.y * difference.y;
  }

  float meanDifference = sqrt(totalsquaredifference) / notp;
  printf("meanDifference: %f\n", meanDifference);
  return meanDifference;
}

__host__ int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    int threadsPerBlock = 256;
    int numberOfTemporalPoints = 1024*2;
    int numberOfPropagationPoints = 100;
    float thickness = 1.0f; // mm
    float nonLinearRefIndex = 2.1e-4f; // cm2/TW
    // float nonLinearRefIndex = 2.1e6f; // cm2/TW
    float linearRefIndex = 1.45;
    float intensity = 10.0f;      // TW/cm2
    float pulseDuration = 30.0f; // fs
    float centralWavelength = 800.0f; // mm
    float timeStep = 0.1f; // fs
    std::string fileName = "output";

    printfCUDAinfo(argc, argv);

    // Read CL arguments
    readArgument("threadsPerBlock", &threadsPerBlock, argc, argv);
    readArgument("notp", &numberOfTemporalPoints, argc, argv);
    readArgument("nopp", &numberOfPropagationPoints, argc, argv);
    readArgument("thickness", &thickness, argc, argv);
    readArgument("intensity", &intensity, argc, argv);
    readArgument("tau", &pulseDuration, argc, argv);
    readArgument("wl0", &centralWavelength, argc, argv);
    readArgument("dt", &timeStep, argc, argv);
    readArgument("output", &fileName, argc, argv);
    printf("\n\n");

    // Define grid and electric field
    float time[numberOfTemporalPoints];
    cuFloatComplex field0[numberOfTemporalPoints];
    cuFloatComplex fieldFinal[numberOfTemporalPoints];
    for (int i = 0; i < numberOfTemporalPoints; i++)
    {
      time[i] = timeStep * i - numberOfTemporalPoints / 2 * timeStep;
      field0[i] = electricField(time[i], pulseDuration, intensity);
      // Filling up the result array with zeros for easier debugging
      fieldFinal[i].x = 0;
      fieldFinal[i].y = 0;
    }


    //Allocate device memory
    cuFloatComplex *d_field0 = allocateDeviceMemory(numberOfTemporalPoints);
    cuFloatComplex *d_fieldFinal = allocateDeviceMemory(numberOfTemporalPoints);

    // Calculate and copy the differential equation's factor into constant memory
    cuFloatComplex diffEqFactor;
    diffEqFactor.x = 0;
    diffEqFactor.y = -1*nonLinearRefIndex * (2 * pi / centralWavelength * 1e6) / linearRefIndex; // in 1/mm unit
    cudaMemcpyToSymbol(d_diffEqFactor, &diffEqFactor, sizeof(cuFloatComplex), 0, cudaMemcpyHostToDevice);

    // Copy arrray from host to device
    copyFromHostToDevice(field0, d_field0, numberOfTemporalPoints);

    // Execute the kernel
    executeKernel(d_field0, d_fieldFinal, threadsPerBlock, numberOfTemporalPoints, numberOfPropagationPoints, thickness, centralWavelength, timeStep);

    // Copy the result back to the host
    copyFromDeviceToHost(d_fieldFinal, fieldFinal, numberOfTemporalPoints);

    // Deallocate memory
    deallocateMemory(d_field0);
    deallocateMemory(d_fieldFinal);

    cleanUpDevice();

    // Calculate the analytical solution
    cuFloatComplex *analytical = analyticalSolution(field0, numberOfTemporalPoints, nonLinearRefIndex, centralWavelength, linearRefIndex, thickness);

    // Calculate the difference between the analytical and numerical solutions
    calculateDifference(fieldFinal, analytical, numberOfTemporalPoints);

    // Save the data for visualisation
    saveData(fileName, time, field0, fieldFinal, analytical, numberOfTemporalPoints);

    free(analytical);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}

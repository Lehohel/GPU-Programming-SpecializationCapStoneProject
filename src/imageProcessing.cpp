#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

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

void add(std::string sFilename, int addNumber){
  // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
      std::cout << "imageProcessing::add opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "imageProcessing::add unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilenameAdd = sFilename;

    std::string::size_type dot = sResultFilenameAdd.rfind('.');

    if (dot != std::string::npos)
    {
      sResultFilenameAdd = sResultFilenameAdd.substr(0, dot);
    }

    sResultFilenameAdd += "_add.pgm";

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

    // run addition filter
    Npp8u adder = addNumber;

    NPP_CHECK_NPP(nppiAddC_8u_C1RSfs(
        oDeviceSrc.data(), oDeviceSrc.pitch(), adder,
        oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, 0));

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilenameAdd, oHostDst);
    std::cout << "Saved image: " << sResultFilenameAdd << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
}

void erode(std::string sFilename){
  // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
      std::cout << "imageProcessing::erode opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "imageProcessing::erode unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilenameAdd = sFilename;

    std::string::size_type dot = sResultFilenameAdd.rfind('.');

    if (dot != std::string::npos)
    {
      sResultFilenameAdd = sResultFilenameAdd.substr(0, dot);
    }

    sResultFilenameAdd += "_erode.pgm";

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);


    // create struct with box-filter mask size
    NppiSize oMaskSize = {3, 3};
    Npp8u mask[9] = {0, 1, 0,
                     1, 1, 1,
                     0, 1, 0};

    Npp8u *dvc_mask;
    cudaMalloc(&dvc_mask, sizeof(mask));
    cudaMemcpy(dvc_mask, mask, sizeof(mask), cudaMemcpyHostToDevice);

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // NppiSize oSizeROI = {width - oMaskSize.width + 1, height - oMaskSize.height + 1};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width-4, oSizeROI.height-4);
    // set anchor point inside the mask to (oMaskSize.width / 2,
    // oMaskSize.height / 2) It should round down when odd
    NppiPoint oAnchor = {1, 1};

    // run erode

    //// run erode
    NPP_CHECK_NPP(nppiErode_8u_C1R(
      oDeviceSrc.data(), oDeviceSrc.pitch(),
      oDeviceDst.data(), oDeviceDst.pitch(),
      oSizeROI, dvc_mask, oMaskSize, oAnchor));

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilenameAdd, oHostDst);
    std::cout << "Saved image: " << sResultFilenameAdd << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;
    char *addNumberChar;
    int addNumber;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "addNumber"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "addNumber", &addNumberChar);
      addNumber = std::stoi(addNumberChar);
    }
    else
    {
      addNumber = 100;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "Lena.pgm";
    }

    add(sFilename, addNumber);
    printf("\n");
    erode(sFilename);
    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
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

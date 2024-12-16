#include <iostream>
#include <cuda_runtime.h>

void printDeviceProperties(const cudaDeviceProp &prop) {
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Memory Pitch: " << prop.memPitch << " bytes" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dimension: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "Max Grid Size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << "Clock Rate: " << prop.clockRate << " kHz" << std::endl;
    std::cout << "Total Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Device Overlap: " << prop.deviceOverlap << std::endl;
    std::cout << "Multi-Processor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Kernel Execution Timeout Enabled: " << prop.kernelExecTimeoutEnabled << std::endl;
    std::cout << "Integrated: " << prop.integrated << std::endl;
    std::cout << "Can Map Host Memory: " << prop.canMapHostMemory << std::endl;
    std::cout << "Compute Mode: " << prop.computeMode << std::endl;
    std::cout << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;
    std::cout << "ECC Enabled: " << prop.ECCEnabled << std::endl;
    std::cout << "PCI Bus ID: " << prop.pciBusID << std::endl;
    std::cout << "PCI Device ID: " << prop.pciDeviceID << std::endl;
    std::cout << "PCI Domain ID: " << prop.pciDomainID << std::endl;
    std::cout << "TCC Driver: " << prop.tccDriver << std::endl;
    std::cout << "Async Engine Count: " << prop.asyncEngineCount << std::endl;
    std::cout << "Unified Addressing: " << prop.unifiedAddressing << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate << " kHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize << " bytes" << std::endl;
    std::cout << "Max Threads per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "Device " << device << " Properties:" << std::endl;
        printDeviceProperties(deviceProp);
        std::cout << std::endl;
    }

    return 0;
}
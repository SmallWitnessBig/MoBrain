//
// Created by 31530 on 2025/8/10.
//

#ifndef MOBRAIN_BASIC_CUH
#define MOBRAIN_BASIC_CUH

#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <thrust/random.h>
#include <random>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include "utils/readfile.hpp"
#include "utils/writefile.hpp"



#define dt 0.01f


typedef std::string Name;
#define CUDA_CHECK(condition) { GPUAssert((condition), __FILE__, __LINE__); }

inline void GPUAssert(const cudaError_t code, const char *file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
void cudaInit();
void randomInitDeviceF(thrust::device_vector<float>& vec, float min_val, float max_val);
void randomInit(std::vector<float>& data, float min_val, float max_val);

#endif //MOBRAIN_BASIC_CUH
//
// Created by 31530 on 2025/8/10.
//

#include "basic.cuh"

void cudaInit() {

}

// 随机初始化主机端向量
void randomInit(std::vector<float>& data, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(min_val, max_val);

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dis(gen);
    }
}
// 用于设备端随机初始化的函子
struct random_functor {
    float min_val;
    float max_val;
    unsigned int seed;

    random_functor(float _min_val, float _max_val, unsigned int _seed = 777)
        : min_val(_min_val), max_val(_max_val), seed(_seed) {}

    __host__ __device__
    float operator()(const unsigned int n) const {
        // 简单的线性同余生成器
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution dist(min_val, max_val);
        rng.discard(n + seed);
        return dist(rng);
    }
};

void randomInitDeviceF(thrust::device_vector<float>& data, float min_val, float max_val) {
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(data.size()),
                      data.begin(),
                      random_functor(min_val, max_val));
}


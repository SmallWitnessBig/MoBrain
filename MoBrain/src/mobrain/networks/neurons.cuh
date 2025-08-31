#pragma once
#pragma warning(disable: 4668)
#pragma warning(disable: 5039)
#pragma warning(disable: 4514)

#include <map>
#include <string>
#include "basic.cuh"
#include "utils/usefulMap.hpp"

// HH模型状态定义
#define STATE_RESTING 0      // 静息状态
#define STATE_ACTIVATION 1   // 激活期
#define STATE_ABSOLUTE_REFRACTORY 2  // 绝对不应期
#define STATE_RELATIVE_REFRACTORY 3  // 相对不应期
#define STATE_SUPERNORMAL 4  // 超常期
#define STATE_SUBNORMAL 5    // 低常期
using dvf = thrust::device_vector<float>;
/* 默认单位：时间    秒(s)
 *         电压    伏(v)
 *         电流    安(A)
*/
enum NeuronType {
    LIF,
    HH
};
class NeuronGroup {
public:
    virtual ~NeuronGroup() = default;
    unsigned int num;
    glm::vec3 pos;
    NeuronType type;
    std::map<std::string, float> params;
    std::map<std::string, dvf> states;

    virtual void run(cudaStream_t stream) = 0;

    // 添加随机初始化方法
    virtual void randomInitStates(const std::string& state_name, float min_val, float max_val) {
        if (states.contains(state_name)) {
            randomInitDeviceF(states[state_name], min_val, max_val);
        }
    }
};



__global__ void LIFrun(
    float *v,
    float *i,
    float *spike,
    float tau_m,
    float v_rest,
    float v_thresh,
    float v_reset,
    float r_m,
    unsigned int num
);

__global__ void HHrun(
    float* v,
    float* m,
    float* n,
    float* h,
    float* i,
    float* spike,
    float* neuron_state,
    float* state_timer,
    float c_m,
    float g_na,
    float g_k,
    float g_l,
    float v_na,
    float v_k,
    float v_l,
    float tau_syn,
    unsigned int num
);

using dvf = thrust::device_vector<float>;

class LIFGroup : public NeuronGroup {
public:
    LIFGroup(glm::vec3 pos,const unsigned int number);
    void run(cudaStream_t stream) override;
};



class HHGroup : public NeuronGroup {
public:
    HHGroup(glm::vec3 pos,const unsigned int number);
    void run(cudaStream_t stream) override;
};


class NeuronGroupPreset {
public:
    Name name = "";
    NeuronType type = LIF;
    int        num  = 128;
    NeuronGroupPreset(Name name) : name(name) {}
    NeuronGroupPreset(int t):name("default"+std::to_string(t)){}
    NeuronGroupPreset(Name name, NeuronType type, int num):name(name),type(type),num(num){}
    NeuronGroupPreset() = default;
};


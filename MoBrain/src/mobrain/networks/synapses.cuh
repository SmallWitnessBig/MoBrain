//
// Created by 31530 on 2025/8/10.
//

#ifndef MOBRAIN_SYNAPSES_CUH
#define MOBRAIN_SYNAPSES_CUH
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <map>
#include <string>
#include <vector>

#include "basic.cuh"
#include "neurons.cuh"

// 突触结构体，定义单个突触连接
struct Synapse {
    size_t pre_idx;   // 前神经元索引
    size_t post_idx;  // 后神经元索引
    float weight;           // 突触权重
};
void randomInitDeviceS(thrust::device_vector<Synapse>& vec, float min_val, float max_val);
enum class SynapseType {
    Static,
    STDP
};
// 突触基类
class SynapseGroup {
public:
    virtual ~SynapseGroup() = default;
    size_t num;  // 突触数量
    std::map<std::string, float> params;
    std::map<std::string, dvf> states;
    NeuronGroup* src = nullptr;  // 源神经元组
    NeuronGroup* dst = nullptr;  // 目标神经元组
    thrust::device_vector<Synapse> synapses;  // 突触连接列表

    virtual void run(cudaStream_t stream) = 0;
    
    // 添加随机初始化方法
    void randomInitWeights(float min_val, float max_val);
    
    void randomInitStates(const std::string& state_name, float min_val, float max_val) ;
    // 添加连接
    virtual void addConnection(const size_t pre_idx, const size_t post_idx, float weight = 0.0f)=0;

    // 批量添加连接
    virtual void addConnections(const std::vector<size_t>& pre_indices,
                       const std::vector<size_t>& post_indices,
                       const std::vector<float>& weights = {})=0;
};

// 静态突触基类
class StaticSynapses : public SynapseGroup {
public:
    StaticSynapses(
        NeuronGroup* source,
        NeuronGroup* destination
        ) {
        src = source;
        dst = destination;
    }
    // 添加连接
    void addConnection(const size_t pre_idx, const size_t post_idx, float weight = -1.0f) override;

    // 批量添加连接
    void addConnections(const std::vector<size_t>& pre_indices,
                       const std::vector<size_t>& post_indices,
                       const std::vector<float>& weights = {}) override;

};

// STDP突触基类
class STDPsynapses : public SynapseGroup {
public:
    STDPsynapses(NeuronGroup* source, NeuronGroup* destination,float weight = 0.5f) {
        src = source;
        dst = destination;
        // STDP参数
        params["a_plus"] = 0.005f;     // LTP幅度参数
        params["a_minus"] = 0.0055f;   // LTD幅度参数
        params["tau_plus"] = 20.0f;    // LTP时间常数
        params["tau_minus"] = 20.0f;   // LTD时间常数
        params["w_max"] = 1.0f;        // 最大权重
        params["w_min"] = 0.0f;        // 最小权重
        params["weight"] = weight;       // 初始权重
    }
    
    // 添加连接
    void addConnection(const size_t pre_idx, const size_t post_idx, float weight = -1.0f) override;

    // 批量添加连接
    void addConnections(const std::vector<size_t>& pre_indices,
                       const std::vector<size_t>& post_indices,
                       const std::vector<float>& weights = {}) override;
};

// LIF神经元专用静态突触
class LIFStaticSynapses : public StaticSynapses {
public:
    LIFStaticSynapses(NeuronGroup* source, NeuronGroup* destination)
        : StaticSynapses(source, destination) {
        // LIF特定参数
        params["delay"] = 1.0f;        // 突触延迟
        params["tau_syn"] = 5.0f;      // 突触时间常数
    }

    void run(cudaStream_t stream) override;  // 支持CUDA流的实现
};

// LIF神经元专用STDP突触
class LIFSTDPSynapses : public STDPsynapses {
public:
    LIFSTDPSynapses(NeuronGroup* source, NeuronGroup* destination);

    void run(cudaStream_t stream) override;  // 支持CUDA流的实现
};

// HH神经元专用静态突触
class HHStaticSynapses : public StaticSynapses {
public:
    HHStaticSynapses(NeuronGroup* source, NeuronGroup* destination)
        : StaticSynapses(source, destination) {
        // HH特定参数
        params["delay"] = 1.0f;        // 突触延迟
        params["tau_syn"] = 5.0f;      // 突触时间常数
        // HH模型中突触传递可能需要额外的参数
        params["e_syn"] = 0.0f;        // 突触反转电位 (mV)
    }

    void run(cudaStream_t stream) override;  // 支持CUDA流的实现
};

// HH神经元专用STDP突触
class HHSTDPSynapses : public STDPsynapses {
public:
    HHSTDPSynapses(NeuronGroup* source, NeuronGroup* destination);

    void run(cudaStream_t stream) override;  // 支持CUDA流的实现
};

class SynapseGroupPreset {
public:
    class connection {
    public:
        Name src;
        Name dst;
    };
    std::vector<connection> connections;
};
class NeuronBlockPreset {
public:
    std::unordered_map<Name,NeuronGroupPreset> presets;
    std::unordered_map<Name,SynapseGroupPreset> synapse_presets;
};
#endif //MOBRAIN_SYNAPSES_CUH
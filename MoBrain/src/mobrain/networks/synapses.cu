//
// Created by 31530 on 2025/8/10.
//

#include "synapses.cuh"
// 用于设备端随机初始化Synapse权重的函子
struct synapse_random_functor {
    float min_val;
    float max_val;
    unsigned int seed;

    synapse_random_functor(float _min_val, float _max_val, unsigned int _seed = 777)
        : min_val(_min_val), max_val(_max_val), seed(_seed) {}

    __host__ __device__
    Synapse operator()(const Synapse& syn) const {
        // 创建一个新的Synapse结构体，保持索引不变，只更新权重
        Synapse new_syn = syn;

        // 简单的线性同余生成器
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(min_val, max_val);
        rng.discard(syn.pre_idx + syn.post_idx + seed);
        new_syn.weight = dist(rng);

        return new_syn;
    }
};

void randomInitDeviceS(thrust::device_vector<Synapse>& data, float min_val, float max_val) {
    thrust::transform(data.begin(),
                      data.end(),
                      data.begin(),
                      synapse_random_functor(min_val, max_val));
}

void SynapseGroup::randomInitWeights(float min_val, float max_val) {
   randomInitDeviceS(synapses, min_val, max_val);
}

void SynapseGroup::randomInitStates(const std::string &state_name, float min_val, float max_val) {
    if (states.contains(state_name)) {
        randomInitDeviceF(states[state_name], min_val, max_val);
    }
}

void StaticSynapses::addConnection(const size_t pre_idx, const size_t post_idx, float weight) {
    const Synapse syn = {pre_idx, post_idx, weight};
    // 在主机端临时存储，稍后复制到设备端
    thrust::host_vector<Synapse> temp(1, syn);
    synapses.insert(synapses.end(), temp.begin(), temp.end());
    num = synapses.size();
}

void StaticSynapses::addConnections(const std::vector<size_t>& pre_indices,
                       const std::vector<size_t>& post_indices,
                       const std::vector<float>& weights) {
    if (pre_indices.size() != post_indices.size()) return;  // 大小不匹配

    thrust::host_vector<Synapse> temp;
    temp.reserve(pre_indices.size());

    for (size_t i = 0; i < pre_indices.size(); ++i) {
        const float weight = weights[i];
        Synapse syn = {pre_indices[i], post_indices[i], weight};
        temp.push_back(syn);
    }

    synapses.insert(synapses.end(), temp.begin(), temp.end());
    num = synapses.size();
}


void STDPsynapses::addConnection(const size_t pre_idx, const size_t post_idx, float weight) {
    const Synapse syn = {pre_idx, post_idx, weight};
    // 在主机端临时存储，稍后复制到设备端
    thrust::host_vector<Synapse> temp(1, syn);
    synapses.insert(synapses.end(), temp.begin(), temp.end());
    num = synapses.size();
}

void STDPsynapses::addConnections(const std::vector<size_t>& pre_indices,
                       const std::vector<size_t>& post_indices,
                       const std::vector<float>& weights ) {
    if (pre_indices.size() != post_indices.size()) return;  // 大小不匹配

    thrust::host_vector<Synapse> temp;
    temp.reserve(pre_indices.size());

    for (size_t i = 0; i < pre_indices.size(); ++i) {
        const float weight = weights[i];
        Synapse syn = {pre_indices[i], post_indices[i], weight};
        temp.push_back(syn);
    }

    synapses.insert(synapses.end(), temp.begin(), temp.end());
    num = synapses.size();
}


LIFSTDPSynapses::LIFSTDPSynapses(NeuronGroup* source, NeuronGroup* destination)
        : STDPsynapses(source, destination) {
    // LIF特定参数
    params["delay"] = 1.0f;        // 突触延迟
    params["tau_syn"] = 5.0f;      // 突触时间常数

    // 初始化STDP状态变量
    states["last_pre_spike"] = thrust::device_vector<float>(src->num, -1.0f);   // 上次前神经元发放时间
    states["last_post_spike"] = thrust::device_vector<float>(dst->num, -1.0f);  // 上次后神经元发放时间
}
HHSTDPSynapses::HHSTDPSynapses(NeuronGroup* source, NeuronGroup* destination)
        : STDPsynapses(source, destination) {
    // HH特定参数
    params["delay"] = 1.0f;        // 突触延迟
    params["tau_syn"] = 5.0f;      // 突触时间常数
    // HH模型中突触传递可能需要额外的参数
    params["e_syn"] = 0.0f;        // 突触反转电位 (mV)

    // 初始化STDP状态变量
    states["last_pre_spike"] = thrust::device_vector<float>(src->num, -1.0f);   // 上次前神经元发放时间
    states["last_post_spike"] = thrust::device_vector<float>(dst->num, -1.0f);  // 上次后神经元发放时间
}
// LIF静态突触CUDA内核
__global__ void LIFStaticSynapsesRun(
    const float *src_spike,
    float *dst_i,
    const Synapse *synapses,
    const size_t num
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;

    const Synapse syn = synapses[idx];
    // 突触电流贡献 = 权重 * 前神经元spike
    const float synaptic_current = syn.weight * src_spike[syn.pre_idx];
    
    // 将突触电流添加到目标神经元的输入电流中
    atomicAdd(&dst_i[syn.post_idx], synaptic_current);
}

// 优化的LIF STDP突触CUDA内核
__global__ void LIFSTDPsynapsesRun(
    const float *src_spike,
    const float *dst_spike,
    float *dst_i,
    Synapse *synapses,
    float *last_pre_spike,
    float *last_post_spike,
    const float a_plus,
    const float a_minus,
    const float tau_plus,
    const float tau_minus,
    const float w_max,
    const float w_min,
    const size_t num,
    const float current_time
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;

    Synapse syn = synapses[idx];
    
    // 使用寄存器缓存常用值
    float pre_spike = src_spike[syn.pre_idx];
    float post_spike = dst_spike[syn.post_idx];
    
    // 计算并添加突触电流 (与静态突触保持一致)
    float synaptic_current = syn.weight * pre_spike;
    atomicAdd(&dst_i[syn.post_idx], synaptic_current);
    
    // 使用寄存器缓存时间戳
    float pre_time = last_pre_spike[syn.pre_idx];
    float post_time = last_post_spike[syn.post_idx];
    
    // 更新spike时间记录（减少全局内存写入）
    if (pre_spike > 0.5f) {
        pre_time = current_time;
    }
    
    if (post_spike > 0.5f) {
        post_time = current_time;
    }
    
    // STDP权重更新（合并条件判断）
    if (pre_time >= 0 && post_time >= 0) {
        const float delta_t = post_time - pre_time;
        float dw = 0.0f;
        
        if (delta_t > 0) {
            // 使用快速指数函数
            dw = a_plus * __expf(-delta_t / tau_plus);
        } else if (delta_t < 0) {
            // 使用快速指数函数
            dw = -a_minus * __expf(delta_t / tau_minus);
        }
        
        // 合并权重更新和裁剪
        float new_weight = syn.weight + dw;
        synapses[idx].weight = fmaxf(fminf(new_weight, w_max), w_min);
    }
    
    // 批量更新全局内存（减少访问次数）
    if (pre_spike > 0.5f) {
        last_pre_spike[syn.pre_idx] = pre_time;
    }
    
    if (post_spike > 0.5f) {
        last_post_spike[syn.post_idx] = post_time;
    }
}

// HH静态突触CUDA内核
__global__ void HHStaticSynapsesRun(
    const float* src_spike,
    float* dst_i,
    const Synapse* synapses,
    float tau_syn,
    float e_syn,  // 突触反转电位
    const size_t num
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    
    Synapse syn = synapses[idx];
    // 突触电流贡献 = 权重 * 前神经元spike * (e_syn - v_post)
    // 这里简化处理，假设spike为1时表示发放，0时表示未发放
    float synaptic_current = syn.weight * src_spike[syn.pre_idx];
    
    // 将突触电流添加到目标神经元的输入电流中
    atomicAdd(&dst_i[syn.post_idx], synaptic_current);
    
    // 电流按指数衰减（在神经元级别处理）
}

// 优化的HH STDP突触CUDA内核
__global__ void HHSTDPsynapsesRun(
    const float *src_spike,
    const float *dst_spike,
    float *dst_i,
    Synapse *synapses,
    float *last_pre_spike,
    float *last_post_spike,
    const float a_plus,
    const float a_minus,
    const float tau_plus,
    const float tau_minus,
    const float w_max,
    const float w_min,
    const size_t num,
    const float current_time,
    const float e_syn,
    const float *neuron_state
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    
    Synapse syn = synapses[idx];
    
    // 使用寄存器缓存常用值
    float pre_spike = src_spike[syn.pre_idx];
    float post_spike = dst_spike[syn.post_idx];
    float post_state = neuron_state[syn.post_idx];
    
    // 只有在后神经元不在不应期时才计算突触电流（合并条件判断）
    if (post_state != STATE_ABSOLUTE_REFRACTORY && post_state != STATE_RELATIVE_REFRACTORY) {
        // 使用快速指数函数
        if (pre_spike > 0.5f) {
            // 简化电流计算公式
            float delta = e_syn - dst_i[syn.post_idx];
            atomicAdd(&dst_i[syn.post_idx], syn.weight * delta);
        }
    }
    
    // 使用寄存器缓存时间戳
    float pre_time = last_pre_spike[syn.pre_idx];
    float post_time = last_post_spike[syn.post_idx];
    
    // 更新spike时间记录（减少全局内存写入）
    if (pre_spike > 0.5f) {
        pre_time = current_time;
    }
    
    if (post_spike > 0.5f) {
        post_time = current_time;
    }
    
    // STDP权重更新（合并条件判断）
    if (pre_time >= 0 && post_time >= 0) {
        const float delta_t = post_time - pre_time;
        float dw = 0.0f;
        
        if (delta_t > 0) {
            // 使用快速指数函数
            dw = a_plus * __expf(-delta_t / tau_plus);
        } else if (delta_t < 0) {
            // 使用快速指数函数
            dw = -a_minus * __expf(delta_t / tau_minus);
        }
        
        // 合并权重更新和裁剪
        float new_weight = syn.weight + dw;
        synapses[idx].weight = fmaxf(fminf(new_weight, w_max), w_min);
    }
    
    // 批量更新全局内存（减少访问次数）
    if (pre_spike > 0.5f) {
        last_pre_spike[syn.pre_idx] = pre_time;
    }
    
    if (post_spike > 0.5f) {
        last_post_spike[syn.post_idx] = post_time;
    }
}

// LIFStaticSynapses实现

void LIFStaticSynapses::run(cudaStream_t stream) {
    // 确保设备向量已分配
    if (synapses.size() != num) {
        synapses.resize(num);
    }

    const float* src_spike = thrust::raw_pointer_cast(src->states["spike"].data());
    float* dst_i = thrust::raw_pointer_cast(dst->states["i"].data());
    const Synapse* syn_ptr = thrust::raw_pointer_cast(synapses.data());
    
    LIFStaticSynapsesRun<<<(num + 255) / 256, 256, 0, stream>>>(
        src_spike,
        dst_i,
        syn_ptr,
        num
    );
}

// LIFSTDPsynapses实现

void LIFSTDPSynapses::run(cudaStream_t stream) {
    // 确保设备向量已分配
    if (synapses.size() != num) {
        synapses.resize(num);
    }

    const float* src_spike = thrust::raw_pointer_cast(src->states["spike"].data());
    const float* dst_spike = thrust::raw_pointer_cast(dst->states["spike"].data());
    float* dst_i = thrust::raw_pointer_cast(dst->states["i"].data());
    Synapse* syn_ptr = thrust::raw_pointer_cast(synapses.data());
    float* last_pre_spike = thrust::raw_pointer_cast(states["last_pre_spike"].data());
    float* last_post_spike = thrust::raw_pointer_cast(states["last_post_spike"].data());

    // 获取当前时间（简化实现，实际应从仿真时间管理器获取）
    static float current_time = 0.0f;
    current_time += dt; // 使用全局时间步长
    
    LIFSTDPsynapsesRun<<<(num + 255) / 256, 256, 0, stream>>>(
        src_spike,
        dst_spike,
        dst_i,
        syn_ptr,
        last_pre_spike,
        last_post_spike,
        params["a_plus"],
        params["a_minus"],
        params["tau_plus"],
        params["tau_minus"],
        params["w_max"],
        params["w_min"],
        num,
        current_time
    );
}

// HHStaticSynapses实现

void HHStaticSynapses::run(cudaStream_t stream) {
    // 确保设备向量已分配
    if (synapses.size() != num) {
        synapses.resize(num);
    }

    const float* src_spike = thrust::raw_pointer_cast(src->states["spike"].data());
    float* dst_i = thrust::raw_pointer_cast(dst->states["i"].data());
    const Synapse* syn_ptr = thrust::raw_pointer_cast(synapses.data());
    
    HHStaticSynapsesRun<<<(num + 255) / 256, 256, 0, stream>>>(
        src_spike,
        dst_i,
        syn_ptr,
        params["tau_syn"],
        params["e_syn"],
        num
    );
}

// HHSTDPsynapses实现

void HHSTDPSynapses::run(cudaStream_t stream) {
    // 确保设备向量已分配
    if (synapses.size() != num) {
        synapses.resize(num);
    }

    const float* src_spike = thrust::raw_pointer_cast(src->states["spike"].data());
    const float* dst_spike = thrust::raw_pointer_cast(dst->states["spike"].data());
    float* dst_i = thrust::raw_pointer_cast(dst->states["i"].data());
    Synapse* syn_ptr = thrust::raw_pointer_cast(synapses.data());
    float* last_pre_spike = thrust::raw_pointer_cast(states["last_pre_spike"].data());
    float* last_post_spike = thrust::raw_pointer_cast(states["last_post_spike"].data());
    float* neuron_state = thrust::raw_pointer_cast(dst->states["neuron_state"].data());
    // 获取神经元状态（新增参数）
    
    // 获取当前时间（简化实现，实际应从仿真时间管理器获取）
    static float current_time = 0.0f;
    current_time += dt; // 使用全局时间步长
    
    HHSTDPsynapsesRun<<<(num + 255) / 256, 256, 0, stream>>>(
        src_spike,
        dst_spike,
        dst_i,
        syn_ptr,
        last_pre_spike,
        last_post_spike,
        params["a_plus"],
        params["a_minus"],
        params["tau_plus"],
        params["tau_minus"],
        params["w_max"],
        params["w_min"],
        num,
        current_time,
        params["e_syn"],
        neuron_state  // 传递神经元状态参数
    );
}

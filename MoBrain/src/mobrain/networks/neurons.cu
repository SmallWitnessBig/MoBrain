#include "neurons.cuh"




__global__ void LIFrun(
    float* v,
    float* i,
    float* spike,
    float tau_m,
    float v_rest,
    float v_thresh,
    float v_reset,
    float r_m,
    unsigned int num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    // 计算 dv = (-(v - v_rest) + R * i) / tau_m
    // 正确实现LIF模型的膜电位更新公式
    float dv = (-(v[idx] - v_rest) + r_m * i[idx]) / tau_m;
    v[idx] += dv * dt;
    
    // 检查是否达到阈值并产生 spike
    if (v[idx] >= v_thresh) {
        v[idx] = v_reset;
        spike[idx] = 1.0f;
    } else {
        spike[idx] = 0.0f;
    }
    
    // 衰减电流
    i[idx] *= exp(-dt / tau_m);
}



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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;
    
    // 使用寄存器缓存频繁访问的状态
    int current_state = neuron_state[idx];
    float current_timer = state_timer[idx];

    
    // 使用临时变量存储计算结果，减少全局内存访问
    float v_val = v[idx];
    float m_val = m[idx];
    float n_val = n[idx];
    float h_val = h[idx];
    float i_val = i[idx];
    float spike_val = 0.0f;
    
    // 优化后的状态处理
    switch(current_state) {
        case STATE_RESTING:
        case STATE_SUPERNORMAL:
        case STATE_SUBNORMAL:
            {
                // HH模型中的离子电流计算
                float i_na = g_na * powf(m_val, 3) * h_val * (v_val - v_na);
                float i_k = g_k * powf(n_val, 4) * (v_val - v_k);
                float i_l = g_l * (v_val - v_l);
                
                // 膜电位更新
                float dv = (i_val - i_na - i_k - i_l) / c_m;
                v_val += dv * dt;

                // 门控变量计算（使用合并后的逻辑）
                float alpha_m = (v_val + 40.0f) / (1.0f - __expf(-(v_val + 40.0f) / 10.0f));
                float beta_m = 4.0f * __expf(- (v_val + 65.0f) / 18.0f);
                float alpha_h = 0.07f * __expf(- (v_val + 65.0f) / 20.0f);
                float beta_h = 1.0f / (1.0f + __expf(- (v_val + 35.0f) / 10.0f));
                float alpha_n = (v_val + 55.0f) / (1.0f - __expf(-(v_val + 55.0f) / 10.0f));
                float beta_n = 0.125f * __expf(- (v_val + 65.0f) / 80.0f);
                
                // 更新门控变量
                m_val += (alpha_m * (1.0f - m_val) - beta_m * m_val) * dt;
                h_val += (alpha_h * (1.0f - h_val) - beta_h * h_val) * dt;
                n_val += (alpha_n * (1.0f - n_val) - beta_n * n_val) * dt;
                
                // 不同状态下的阈值调整
                float threshold = (current_state == STATE_SUPERNORMAL) ? 20.0f : 
                                 (current_state == STATE_SUBNORMAL) ? 40.0f : 30.0f;
                
                // 检查spike
                if (v_val >= threshold) {
                    spike_val = 1.0f;
                    current_state = STATE_ACTIVATION;
                    current_timer = 0;
                }
                
                // 状态转换逻辑
                if (current_state == STATE_SUPERNORMAL || current_state == STATE_SUBNORMAL) {
                    current_timer++;
                    if (current_timer >= 20) {
                        current_state = STATE_RESTING;
                        current_timer = 0;
                    }
                }
            }
            break;
            
        case STATE_ACTIVATION:
            // 快速去极化（减少计算量）
            v_val += 100.0f * dt;
            current_timer++;
            if (current_timer >= 10) {
                current_state = STATE_ABSOLUTE_REFRACTORY;
                current_timer = 0;
            }
            spike_val = 1.0f;
            break;
            
        case STATE_ABSOLUTE_REFRACTORY:
            // 快速复极化（简化计算）
            v_val = -70.0f + 20.0f * __expf(-current_timer * dt / 1.0f);
            current_timer++;
            if (current_timer >= 20) {
                current_state = STATE_RELATIVE_REFRACTORY;
                current_timer = 0;
            }
            spike_val = 0.0f;
            break;
            
        case STATE_RELATIVE_REFRACTORY:
            // 降低计算精度的快速处理
            v_val = -75.0f + 5.0f * __expf(-current_timer * dt / 2.0f);
            
            // 使用更简单的指数函数
            current_timer++;
            if (v_val >= 50.0f) {
                spike_val = 1.0f;
                current_state = STATE_ACTIVATION;
                current_timer = 0;
            } else {
                spike_val = 0.0f;
            }
            
            if (current_timer >= 30) {
                current_state = STATE_SUPERNORMAL;
                current_timer = 0;
            }
            break;
    }
    
    // 批量更新全局内存（减少访问次数）
    v[idx] = v_val;
    m[idx] = m_val;
    n[idx] = n_val;
    h[idx] = h_val;
    i[idx] *= expf(-dt / tau_syn);
    spike[idx] = spike_val;
    state_timer[idx] = current_timer;
    neuron_state[idx] = current_state;
}
LIFGroup::LIFGroup(glm::vec3 pos,const unsigned int number) {
    num = number;
    type = LIF;
    this->pos = pos;
    params["v_thresh"] = -50.0f;
    params["v_reset"] = -70.0f;
    params["v_rest"] = -60.0f;
    params["R"] = 5.0f;
    params["C"] = 1e-3;
    params["tau_m"] = params["R"] * params["C"];
    params["tau_syn"] = 5.0f; // 添加突触时间常数
    params["dt"] = 1e-3f; // 添加时间步长参数

    states["v"] = dvf(num, params["v_rest"]);
    states["i"] = dvf(num, 0.0f);
    states["spike"] = dvf(num, 0.0f);
}

void LIFGroup::run(cudaStream_t stream)  {
    float *v = thrust::raw_pointer_cast(states["v"].data());
    float *i = thrust::raw_pointer_cast(states["i"].data());
    float *spike = thrust::raw_pointer_cast(states["spike"].data());
    LIFrun<<<(num+255)/256,256,0, stream>>>(
        v, i, spike,
        params["tau_m"], params["v_rest"],
        params["v_thresh"], params["v_reset"],
        params["R"],
        num);
}

HHGroup::HHGroup(glm::vec3 pos,const unsigned int number) {
    num = number;
    this->pos = pos;
    type = HH;
    // HH模型参数设置（符合原始Hodgkin-Huxley模型参数）
    params["c_m"] = 1.0f; // 膜电容 (uF/cm^2)
    params["g_na"] = 120.0f; // 钠电导 (mS/cm^2)
    params["g_k"] = 36.0f; // 钾电导 (mS/cm^2)
    params["g_l"] = 0.3f; // 漏电导 (mS/cm^2)
    params["v_na"] = 50.0f; // 钠平衡电位 (mV)
    params["v_k"] = -77.0f; // 钾平衡电位 (mV)
    params["v_l"] = -54.387f; // 漏平衡电位 (mV)
    params["tau_syn"] = 5.0f; // 突触时间常数 (ms)

    // 初始化状态变量（使用更符合生物实际的初始值）
    states["v"] = dvf(num, -65.0f); // 初始膜电位 (mV)
    states["m"] = dvf(num, 0.05f); // 钠激活门控变量（初始值接近0）
    states["n"] = dvf(num, 0.3f); // 钾激活门控变量（初始值约0.3）
    states["h"] = dvf(num, 0.6f); // 钠失活门控变量（初始值约0.6）
    states["i"] = dvf(num, 0.0f); // 输入电流
    states["spike"] = dvf(num, 0.0f); // spike输出
    states["state"] = dvf(num, STATE_RESTING);
    states["timer"] = dvf(num, 0);
}

void HHGroup::run(cudaStream_t stream) {
    float *v = thrust::raw_pointer_cast(states["v"].data());
    float *m = thrust::raw_pointer_cast(states["m"].data());
    float *n = thrust::raw_pointer_cast(states["n"].data());
    float *h = thrust::raw_pointer_cast(states["h"].data());
    float *i = thrust::raw_pointer_cast(states["i"].data());
    float *spike = thrust::raw_pointer_cast(states["spike"].data());
    float *neuron_state = thrust::raw_pointer_cast(states["state"].data());
    float *state_timer = thrust::raw_pointer_cast(states["timer"].data());
    HHrun<<<(num+255)/256,256,0, stream>>>(
        v, m, n, h, i, spike,
        neuron_state, state_timer,
        params["c_m"], params["g_na"], params["g_k"], params["g_l"],
        params["v_na"], params["v_k"], params["v_l"], params["tau_syn"],
        num
    );
}



//
// Created by 31530 on 2025/8/27.
//

#include "synapseGui.hpp"

namespace gui {
    
    // 与NeuronGui类似的创建UI方法
    void SynapseGui::draw() {
        ImGui::PushID(this);
        ImGui::Begin("Create Synapse");
        
        // 输入源Neuron ID - 类似NeuronGui的ID生成方式
        ImGui::InputText("Source ID", _src_id,100);
        // 输入目标Neuron ID
        ImGui::InputText("Target ID", _dst_id,100);
        
        // 选择突触类型
        const char* synapse_types[] = { "Static", "STDP" };
        ImGui::Combo("Type", &currentType, synapse_types, 2);
        
        // 创建按钮 - 与NeuronGui保持一致的交互方式
        if (ImGui::Button("Create Synapse")) {

            auto& n = app.net;
            auto src_id = ID(_src_id);
            auto dst_id = ID(_dst_id);
            auto src = n.neuron_map[src_id.pos][src_id.name];
            auto dst = n.neuron_map[dst_id.pos][dst_id.name];
            if (src_id==dst_id) {
                Error("Source and destination ID cannot be the same");
            }else if (src->type!=dst->type) {
                Error("Neuron types must be the same");
            }else if (src->type==LIF&&currentType==0) {
                auto synapse = new LIFStaticSynapses(src, dst);
                n.addSynapseGroup(synapse);
            }
        }
        
        ImGui::End();
        ImGui::PopID();
    }
} // namespace gui

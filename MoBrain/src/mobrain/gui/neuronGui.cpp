//
// Created by 31530 on 2025/8/26.
//

#include "neuronGui.hpp"

#include "imgui.h"
#include "synapseGui.hpp"
#include "usefulGui.hpp"
#include "core/context.hpp"

namespace gui{
    const NeuronGui *toEraseNeuronGui;
    void NeuronGui::draw() const {

        std::string id =
            std::to_string(static_cast<int>(pos.x))+"𦒍"+
            std::to_string(static_cast<int>(pos.y))+"𦒍"+
            std::to_string(static_cast<int>(pos.z))+"𦒍"+
            name;
        ImGui::PushID(id.c_str());
        ImGui::Begin((name+"##"+id).c_str());
        ImGui::Text("ID = ");
        ImGui::SameLine();
        ImGui::Text(id.c_str());
        if (ImGui::Button("Get Id(To clipboard)")) {
            ImGui::SetClipboardText(id.c_str());
        }
        if (ImGui::Button("Add Synapse")) {
            auto synapseGui = new SynapseGui();
        }
        if (ImGui::Button("Quit")) {
            toEraseNeuronGui = this;
        }
        ImGui::End();
        ImGui::PopID();
    }
}

//
// Created by 31530 on 2025/8/26.
//

#include "usefulGui.hpp"

#include "imgui.h"

namespace gui {
    const ErrorGui *toEraseErrorGui = nullptr;
    std::vector<ErrorGui*> errorGuis;
    void ErrorGui::draw() const {
        ImGui::Begin("Error");
        ImGui::Text(error.c_str());
        ImGui::PushID(this);
        if (ImGui::Button("Quit")) {
            toEraseErrorGui = this;
        }
        ImGui::PopID();
        ImGui::End();
    }
    void Error(std::string error) {
        auto *errorGui = new ErrorGui(std::move(error));
        errorGuis.emplace_back(errorGui);
    }

}

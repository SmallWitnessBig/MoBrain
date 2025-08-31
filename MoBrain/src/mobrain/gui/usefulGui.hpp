//
// Created by 31530 on 2025/8/26.
//

#ifndef MOBRAIN_USEFULGUI_HPP
#define MOBRAIN_USEFULGUI_HPP
#include <string>
#include <vector>

#include "imgui.h"


namespace gui {
    inline void right_align() {

        const float window_width = ImGui::GetWindowWidth();
        constexpr float button_width = 200.0f;
        float right_x = window_width - button_width * 2 - ImGui::GetStyle().ItemSpacing.x;
        ImGui::SetCursorPosX(right_x);
    }

    class ErrorGui {
    public:
        const std::string error;
        explicit ErrorGui(std::string error):error(std::move(error)){}
        void draw() const;
    };
    extern const ErrorGui *toEraseErrorGui;
    void Error(std::string error);
    extern std::vector<ErrorGui*> errorGuis;
}


#endif //MOBRAIN_USEFULGUI_HPP
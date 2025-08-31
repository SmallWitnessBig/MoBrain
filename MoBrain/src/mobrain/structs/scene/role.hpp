//
// Created by 31530 on 2025/8/13.
//

#ifndef MOBRAIN_ROLE_HPP
#define MOBRAIN_ROLE_HPP
#include "../engine/graphics.hpp"
#include "networks/neurons.cuh"
class Role {
private:
    double lastLeftClickTime = 0.0;
    double lastRightClickTime = 0.0;
    double lastCClickTime = 0.0;
    const double clickCooldown = 0.2; // 0.1秒冷却时间

public:
    Role();
    ~Role() = default;
    const Cube* focusCube;
    NeuronGroup* focusNeuronGroup;
    Cube* place;
    Role& update();
    void Key();
    void MouseButton();
};


#endif //MOBRAIN_ROLE_HPP
//
// Created by 31530 on 2025/8/26.
//

#ifndef MOBRAIN_NEURONGUI_HPP
#define MOBRAIN_NEURONGUI_HPP
#include <utility>

#include "glm/vec3.hpp"
#include "networks/basic.cuh"

namespace gui{
    class NeuronGui {
    public:
        Name name;
        glm::vec3 pos;
        NeuronGui(glm::vec3 pos,Name name):name(std::move(name)),pos(pos){}
        void draw() const;
    };
    extern const NeuronGui *toEraseNeuronGui;

}



#endif //MOBRAIN_NEURONGUI_HPP
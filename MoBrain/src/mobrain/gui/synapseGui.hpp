#ifndef MOBRAIN_SYNAPSEGUI_HPP
#define MOBRAIN_SYNAPSEGUI_HPP
#include <string>

#include "core/context.hpp"
#include "glm/vec3.hpp"
#include "networks/basic.cuh"
#include "networks/synapses.cuh"
#include "utils/ID.hpp"


namespace gui {
    class SynapseGui {
    public:
        char _src_id[100];
        char _dst_id[100];
        int currentType;
        SynapseType type;
        SynapseGui() = default;
        // 绘制突触连接
        void draw() ;

    };
}


#endif //MOBRAIN_SYNAPSEGUI_HPP
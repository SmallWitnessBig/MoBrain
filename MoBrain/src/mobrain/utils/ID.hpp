//
// Created by 31530 on 2025/8/27.
//

#ifndef MOBRAIN_ID_HPP
#define MOBRAIN_ID_HPP
#include "glm/vec3.hpp"
#include "gui/usefulGui.hpp"
#include "networks/basic.cuh"

class ID {
public:
    glm::vec3 pos;
    Name name;
    std::string _id;
    char id[100];
    explicit ID(std::string id) : _id(std::move(id)){
        _id.reserve(100);
        const size_t x_pos = id.find("𦒍");
        const size_t y_pos = id.find("𦒍", x_pos + 1);
        const size_t z_pos = id.find("𦒍", y_pos + 1);
        if (x_pos == std::string::npos || y_pos == std::string::npos || z_pos == std::string::npos) {
            gui::Error("Invalid ID");
            assert("Invalid ID");
        }
        const float x = std::stof(id.substr(0, x_pos));
        const float y = std::stof(id.substr(x_pos+1, y_pos));
        const float z = std::stof(id.substr(y_pos+1, z_pos));
        this->pos = glm::vec3(x, y, z);
        this->name = id.substr(z_pos+1);
        strcpy_s(this->id, _id.c_str());
    };
    ID(const glm::vec3 pos, const Name &name) : pos(pos), name(name){
        _id = std::to_string(static_cast<int>(pos.x)) + "𦒍" +
         std::to_string(static_cast<int>(pos.y)) + "𦒍" +
         std::to_string(static_cast<int>(pos.z)) + "𦒍" +
         name;
        strcpy_s(id, _id.c_str());
    };

    bool operator==(const ID & id) const = default;
};
#endif //MOBRAIN_ID_HPP
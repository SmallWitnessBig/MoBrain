//
// Created by 31530 on 2025/8/13.
//

#ifndef MOBRAIN_SCENE_HPP
#define MOBRAIN_SCENE_HPP
#include <unordered_map>

#include "../engine/graphics.hpp"
#include "utils/usefulMap.hpp"


class Scene {
public:
    std::vector<const Cube*> cubes;

    void addCube(const Cube* cube);
    void removeCube(const Cube* cube);

    void initScene();
    void writeIntoFile();
    Scene() = default;
    ~Scene();
    std::unordered_map<const Cube*, size_t> cube_map;
    posMap<size_t> pos_map;
};


#endif //MOBRAIN_SCENE_HPP